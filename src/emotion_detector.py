import torch
from torch import nn
from pathlib import Path
import cv2
import numpy as np
import logging
import subprocess

from emonet import EmoNet
from face_alignment.detection.sfd.sfd_detector import SFDDetector


logger = logging.getLogger(__name__)


class EmotionDetector:
    """
    Wrapper class for EmoNet model that handles emotion detection from images.
    Automatically selects the least loaded GPU on initialization.
    """

    emotion_classes = {
        0: "Neutral",
        1: "Happy",
        2: "Sad",
        3: "Surprise",
        4: "Fear",
        5: "Disgust",
        6: "Anger",
        7: "Contempt",
    }

    def __init__(self, n_classes=8, image_size=256):
        """
        Initialize the emotion detector.

        Args:
            n_classes (int): Number of emotion classes (5 or 8)
            image_size (int): Size to resize images to (default: 256)
        """
        self.n_classes = n_classes
        self.image_size = image_size

        # Select GPU with least memory usage
        self.device = self._select_best_gpu()
        logger.info(f"Selected device: {self.device}")

        # Load models
        self.model = self._load_model()
        self.face_detector = self._load_face_detector()

    def _get_gpu_memory_usage(self):
        """Get current GPU memory usage for all GPUs."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            gpu_info = []
            for line in result.stdout.strip().split("\n"):
                parts = line.split(", ")
                if len(parts) == 3:
                    gpu_info.append(
                        {
                            "index": int(parts[0]),
                            "used": int(parts[1]),
                            "total": int(parts[2]),
                            "free": int(parts[2]) - int(parts[1]),
                        }
                    )
            return gpu_info
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return []

    def _select_best_gpu(self):
        """Select GPU with most free memory."""
        gpu_info = self._get_gpu_memory_usage()

        if not gpu_info:
            logger.warning("No GPU info available, using CPU")
            return torch.device("cpu")

        # Sort by free memory (descending)
        gpu_info.sort(key=lambda x: x["free"], reverse=True)

        best_gpu = gpu_info[0]
        logger.info(
            f"GPU {best_gpu['index']} selected: {best_gpu['free']}MB free out of {best_gpu['total']}MB"
        )

        return torch.device(f'cuda:{best_gpu["index"]}')

    def _load_model(self):
        """Load the pretrained EmoNet model."""
        # Construct path to pretrained model
        state_dict_path = (
            Path(__file__)
            .parents[1]
            .joinpath("pretrained", f"emonet_{self.n_classes}.pth")
        )

        if not state_dict_path.exists():
            raise FileNotFoundError(f"Model weights not found at {state_dict_path}")

        logger.info(f"Loading model from {state_dict_path}")

        # Load state dict
        state_dict = torch.load(str(state_dict_path), map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Initialize model
        model = EmoNet(n_expression=self.n_classes).to(self.device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Enable cudnn benchmark for better performance
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        return model

    def _load_face_detector(self):
        """Load the SFD face detector."""
        logger.info("Loading SFD face detector")
        device_str = str(self.device)
        detector = SFDDetector(device_str)
        return detector

    def preprocess_image(self, image):
        """
        Preprocess image for the model.

        Args:
            image: numpy array (BGR format from OpenCV) or RGB array

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Ensure we have RGB format
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]

        # Resize to model input size
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Convert to tensor and normalize to [0, 1]
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        return image_tensor

    def predict(self, image):
        """
        Predict emotion, valence, and arousal from an image.

        Args:
            image: numpy array representing the image (RGB format)

        Returns:
            dict: Contains 'emotion', 'emotion_class', 'valence', 'arousal', 'emotion_probabilities',
                  and 'face_detected' flag
        """
        # Detect faces first
        # SFD detector expects BGR format
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            detected_faces = self.face_detector.detect_from_image(image_bgr)

        # If no face detected, return None values
        if len(detected_faces) == 0:
            logger.warning("No face detected in image")
            return {
                "emotion": None,
                "emotion_class": None,
                "valence": None,
                "arousal": None,
                "emotion_probabilities": None,
                "face_detected": False,
                "message": "No face detected in image",
            }

        # Use the first detected face
        bbox = np.array(detected_faces[0]).astype(np.int32)

        # Ensure bbox is within image bounds
        bbox[0] = max(0, bbox[0])  # x1
        bbox[1] = max(0, bbox[1])  # y1
        bbox[2] = min(image.shape[1], bbox[2])  # x2
        bbox[3] = min(image.shape[0], bbox[3])  # y2

        # Extract face crop (using RGB image)
        face_crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2], :]

        # Check if face crop is valid
        if face_crop.size == 0:
            logger.warning("Invalid face crop dimensions")
            return {
                "emotion": None,
                "emotion_class": None,
                "valence": None,
                "arousal": None,
                "emotion_probabilities": None,
                "face_detected": False,
                "message": "Invalid face crop",
            }

        # Preprocess the face crop
        image_tensor = self.preprocess_image(face_crop)

        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)

            # Get emotion probabilities and prediction
            emotion_probs = nn.functional.softmax(output["expression"], dim=1)
            predicted_class = torch.argmax(emotion_probs).cpu().item()

            # Get valence and arousal, clamped to [-1, 1]
            valence = output["valence"].clamp(-1.0, 1.0).cpu().item()
            arousal = output["arousal"].clamp(-1.0, 1.0).cpu().item()

            # Get all emotion probabilities
            emotion_probs_dict = {
                self.emotion_classes[i]: float(emotion_probs[0, i].cpu())
                for i in range(self.n_classes)
            }

        return {
            "emotion": self.emotion_classes[predicted_class],
            "emotion_class": predicted_class,
            "valence": float(valence),
            "arousal": float(arousal),
            "emotion_probabilities": emotion_probs_dict,
            "face_detected": True,
            "face_bbox": bbox.tolist(),  # Include face bounding box in response
        }

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "face_detector"):
            del self.face_detector
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
