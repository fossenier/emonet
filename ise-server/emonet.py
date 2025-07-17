"""
EmoNet: Emotion Recognition from Facial Features
===============================================

A modern implementation of EmoNet for facial emotion recognition using
hourglass networks with attention mechanisms.

Authors: Jean Kossaifi, Antoine Toisoul, Adrian Bulat
Refactored (Claude Opus 4) for modern PyTorch best practices.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        use_relu: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_relu = use_relu

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.use_relu:
            x = F.relu(x, inplace=True)
        return x


class ConvBlock(nn.Module):
    """
    Convolutional block with multi-scale feature extraction.

    Uses a split-concat architecture to capture features at different scales.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Calculate channel splits
        mid_channels = out_channels // 2
        quarter_channels = out_channels // 4

        # Multi-scale convolution branches
        self.branch1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        )

        self.branch2 = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, quarter_channels, kernel_size=3, padding=1, bias=False
            ),
        )

        self.branch3 = nn.Sequential(
            nn.BatchNorm2d(quarter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                quarter_channels, quarter_channels, kernel_size=3, padding=1, bias=False
            ),
        )

        # Residual connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        # Multi-scale feature extraction
        out1 = self.branch1(x)
        out2 = self.branch2(out1)
        out3 = self.branch3(out2)

        # Concatenate multi-scale features
        out = torch.cat([out1, out2, out3], dim=1)

        # Apply residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        return out


class HourGlass(nn.Module):
    """
    Hourglass module for hierarchical feature extraction.

    Processes features at multiple resolutions to capture both
    local details and global context.
    """

    def __init__(self, depth: int, num_features: int = 256):
        super().__init__()
        self.depth = depth
        self.num_features = num_features

        # Build the hourglass recursively
        self._build_hourglass(depth)

    def _build_hourglass(self, depth: int):
        """Recursively build hourglass layers."""
        # Upper branch (high resolution)
        self.add_module(f"up1_{depth}", ConvBlock(self.num_features, self.num_features))

        # Lower branch (low resolution)
        self.add_module(
            f"low1_{depth}", ConvBlock(self.num_features, self.num_features)
        )

        if depth > 1:
            # Recursive hourglass
            self._build_hourglass(depth - 1)
        else:
            # Bottom layer
            self.add_module(
                f"low2_{depth}", ConvBlock(self.num_features, self.num_features)
            )

        # Upsampling branch
        self.add_module(
            f"low3_{depth}", ConvBlock(self.num_features, self.num_features)
        )

    def _forward_hourglass(self, depth: int, x: Tensor) -> Tensor:
        """Forward pass through hourglass at given depth."""
        # Upper branch (skip connection)
        up1 = self._modules[f"up1_{depth}"](x)  # type: ignore

        # Lower branch with pooling
        low1 = F.max_pool2d(x, kernel_size=2, stride=2)
        low1 = self._modules[f"low1_{depth}"](low1)  # type: ignore

        if depth > 1:
            # Recursive call
            low2 = self._forward_hourglass(depth - 1, low1)
        else:
            # Bottom of hourglass
            low2 = self._modules[f"low2_{depth}"](low1)  # type: ignore

        # Process lower features
        low3 = self._modules[f"low3_{depth}"](low2)  # type: ignore

        # Upsample and combine
        up2 = F.interpolate(low3, scale_factor=2, mode="nearest")

        return up1 + up2

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_hourglass(self.depth, x)


class EmoNet(nn.Module):
    """
    EmoNet: Multi-task emotion recognition network.

    Predicts discrete emotions (classification) and continuous
    valence-arousal values (regression) from facial images.

    Args:
        num_modules: Number of stacked hourglass modules
        n_expression: Number of discrete emotion classes
        n_reg: Number of regression outputs (valence, arousal)
        n_blocks: Number of blocks in emotion feature extractor
        attention: Whether to use attention mechanism
        temporal_smoothing: Whether to apply temporal smoothing
    """

    def __init__(
        self,
        num_modules: int = 2,
        n_expression: int = 8,
        n_reg: int = 2,
        n_blocks: int = 4,
        attention: bool = True,
        temporal_smoothing: bool = False,
    ):
        super().__init__()

        self.num_modules = num_modules
        self.n_expression = n_expression
        self.n_reg = n_reg
        self.attention = attention
        self.temporal_smoothing = temporal_smoothing

        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Feature encoding blocks
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Hourglass modules
        self.hourglass_modules = nn.ModuleList()
        self.hourglass_tops = nn.ModuleList()
        self.hourglass_finals = nn.ModuleList()
        self.hourglass_bns = nn.ModuleList()
        self.hourglass_landmarks = nn.ModuleList()

        # Inter-hourglass connections
        self.hourglass_intermediates = nn.ModuleList()
        self.hourglass_landmarks_intermediates = nn.ModuleList()

        for i in range(num_modules):
            # Hourglass and processing layers
            self.hourglass_modules.append(HourGlass(depth=4, num_features=256))
            self.hourglass_tops.append(ConvBlock(256, 256))
            self.hourglass_finals.append(nn.Conv2d(256, 256, kernel_size=1, bias=False))
            self.hourglass_bns.append(nn.BatchNorm2d(256))
            self.hourglass_landmarks.append(
                nn.Conv2d(256, 68, kernel_size=1, bias=True)
            )

            # Intermediate connections (except for last module)
            if i < num_modules - 1:
                self.hourglass_intermediates.append(
                    nn.Conv2d(256, 256, kernel_size=1, bias=False)
                )
                self.hourglass_landmarks_intermediates.append(
                    nn.Conv2d(68, 256, kernel_size=1, bias=False)
                )

        # Emotion recognition head
        if attention:
            emotion_in_features = 256 * (num_modules + 1)
        else:
            emotion_in_features = 256 * (num_modules + 1) + 68

        # Emotion feature extractor
        self.emotion_encoder = self._build_emotion_encoder(
            emotion_in_features, n_blocks
        )

        # Final classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, n_expression + n_reg),
        )

        # Temporal smoothing setup
        if temporal_smoothing:
            self._setup_temporal_smoothing()

        # Freeze backbone weights (transfer learning)
        self._freeze_backbone()

    def _build_emotion_encoder(self, in_features: int, n_blocks: int) -> nn.Module:
        """Build emotion feature extraction network."""
        layers = [
            nn.Conv2d(in_features, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ]

        for _ in range(n_blocks):
            layers.extend([ConvBlock(256, 256), nn.MaxPool2d(kernel_size=2, stride=2)])

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        return nn.Sequential(*layers)

    def _setup_temporal_smoothing(self):
        """Initialize temporal smoothing components."""
        self.n_temporal_states = 5
        self.temporal_weights = nn.Parameter(
            torch.tensor([0.1, 0.1, 0.15, 0.25, 0.4]).view(1, 5, 1), requires_grad=False
        )
        self.register_buffer(
            "temporal_state",
            torch.zeros(1, self.n_temporal_states, self.n_expression + self.n_reg),
        )
        self._temporal_initialized = False

    def _freeze_backbone(self):
        """Freeze backbone parameters (hourglass modules)."""
        # Freeze initial layers and hourglass modules
        for module in [
            self.initial_conv,
            self.conv2,
            self.conv3,
            self.conv4,
            self.hourglass_modules,
            self.hourglass_tops,
            self.hourglass_finals,
            self.hourglass_bns,
            self.hourglass_landmarks,
            self.hourglass_intermediates,
            self.hourglass_landmarks_intermediates,
        ]:
            for param in module.parameters():
                param.requires_grad = False

    def reset_temporal_state(self, batch_size: int):
        """Reset temporal state for new sequence."""
        if self.temporal_smoothing:
            device = next(self.parameters()).device
            self.temporal_state = torch.zeros(
                batch_size,
                self.n_temporal_states,
                self.n_expression + self.n_reg,
                device=device,
            )
            self._temporal_initialized = True

    def forward(self, x: Tensor, reset_smoothing: bool = False) -> Dict[str, Tensor]:
        """
        Forward pass through EmoNet.

        Args:
            x: Input tensor of shape (B, 3, H, W)
            reset_smoothing: Whether to reset temporal smoothing state

        Returns:
            Dictionary containing:
                - heatmap: Facial landmark heatmaps (B, 68, H/4, W/4)
                - expression: Discrete emotion logits (B, n_expression)
                - valence: Valence predictions (B, 1)
                - arousal: Arousal predictions (B, 1)
        """
        batch_size = x.size(0)

        # Handle temporal smoothing initialization
        if self.temporal_smoothing:
            if not self._temporal_initialized or reset_smoothing:
                self.reset_temporal_state(batch_size)

        # Initial feature extraction
        x = self.initial_conv(x)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        # Store initial features
        encoded_features = x
        hourglass_features = []

        # Process through hourglass modules
        for i in range(self.num_modules):
            # Hourglass processing
            hg_out = self.hourglass_modules[i](encoded_features)
            hg_out = self.hourglass_tops[i](hg_out)
            hg_out = self.hourglass_finals[i](hg_out)
            hg_out = F.relu(self.hourglass_bns[i](hg_out), inplace=True)

            # Landmark prediction
            landmarks = self.hourglass_landmarks[i](hg_out)

            # Inter-hourglass connections
            if i < self.num_modules - 1:
                encoded_features = (
                    encoded_features
                    + self.hourglass_intermediates[i](hg_out)
                    + self.hourglass_landmarks_intermediates[i](landmarks)
                )

            hourglass_features.append(hg_out)

        # Concatenate all hourglass features
        combined_features = torch.cat(hourglass_features, dim=1)

        # Apply attention if enabled
        if self.attention:
            # Use landmarks as attention mask
            attention_mask = torch.sum(landmarks, dim=1, keepdim=True)  # type: ignore
            attention_mask = torch.sigmoid(attention_mask)
            combined_features = combined_features * attention_mask
            emotion_features = torch.cat([x, combined_features], dim=1)
        else:
            emotion_features = torch.cat([x, combined_features, landmarks], dim=1)  # type: ignore

        # Extract emotion features
        emotion_features = self.emotion_encoder(emotion_features)
        emotion_features = emotion_features.view(batch_size, -1)

        # Final predictions
        predictions = self.emotion_classifier(emotion_features)

        # Apply temporal smoothing if enabled
        if self.temporal_smoothing and self.training:
            with torch.no_grad():
                # Shift temporal states
                self.temporal_state[:, :-1] = self.temporal_state[:, 1:].clone()
                self.temporal_state[:, -1] = predictions

                # Weighted average
                predictions = torch.sum(
                    self.temporal_weights * self.temporal_state, dim=1
                )

        # Split predictions
        expression_logits = predictions[:, : self.n_expression]
        valence = predictions[:, self.n_expression]
        arousal = predictions[:, self.n_expression + 1]

        return {
            "heatmap": landmarks,  # type: ignore
            "expression": expression_logits,
            "valence": valence,
            "arousal": arousal,
        }

    def extract_features(self, x: Tensor) -> Tensor:
        """
        Extract intermediate features for analysis or transfer learning.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Feature tensor of shape (B, 256)
        """
        with torch.no_grad():
            # Initial feature extraction
            x = self.initial_conv(x)
            x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
            x = self.conv3(x)
            x = self.conv4(x)

            # Process through first hourglass
            hg_out = self.hourglass_modules[0](x)
            hg_out = self.hourglass_tops[0](hg_out)

            # Global average pooling
            features = F.adaptive_avg_pool2d(hg_out, (1, 1))
            features = features.view(features.size(0), -1)

            return features


def create_emonet(
    pretrained: bool = True, checkpoint_path: Optional[str] = None, **kwargs
) -> EmoNet:
    """
    Create an EmoNet model instance.

    Args:
        pretrained: Whether to load pretrained weights
        checkpoint_path: Path to checkpoint file
        **kwargs: Additional arguments for EmoNet

    Returns:
        EmoNet model instance
    """
    model = EmoNet(**kwargs)

    if pretrained and checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")

    return model


if __name__ == "__main__":
    # Test the model
    model = create_emonet(pretrained=False, n_expression=8)
    model.eval()

    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)

    # Forward pass
    with torch.no_grad():
        outputs = model(x)

    # Check outputs
    print("Model output shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # Check feature extraction
    features = model.extract_features(x)
    print(f"\nExtracted features shape: {features.shape}")

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
