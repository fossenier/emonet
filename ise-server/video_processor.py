"""
EmoNet Video Processor with Batch Processing
==========================================

This module processes videos to detect faces and predict emotions using EmoNet.
Optimized for A100 GPU with batch processing for improved performance.

Usage:
    python emonet_video_processor.py --video_path input.mp4 --output_path output.mp4
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import logging
from contextlib import contextmanager
import time

import numpy as np
import torch
from torch import nn
import cv2
from tqdm import tqdm

from face_alignment.detection.sfd.sfd_detector import SFDDetector
from emonet import EmoNet


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for video processing."""

    n_expression: int = 8
    device: str = "cuda:0"
    image_size: int = 256
    batch_size: int = 32  # Optimized for A100
    video_fps: float = 24.0
    max_faces_per_frame: int = 1

    # Emotion class mapping
    emotion_classes: Dict[int, str] = None  # type: ignore

    def __post_init__(self):
        if self.emotion_classes is None:
            self.emotion_classes = {
                0: "Neutral",
                1: "Happy",
                2: "Sad",
                3: "Surprise",
                4: "Fear",
                5: "Disgust",
                6: "Anger",
                7: "Contempt",
            }


class VideoProcessor:
    """Handles video loading and saving operations."""

    def __init__(self, config: Config):
        self.config = config

    def load_video(self, video_path: Path) -> Tuple[List[np.ndarray], float]:
        """
        Load video frames and extract metadata.

        Args:
            video_path: Path to input video

        Returns:
            Tuple of (list of RGB frames, original FPS)
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Loading video: {video_path}")
        logger.info(f"FPS: {fps}, Total frames: {total_frames}")

        frames = []

        with tqdm(total=total_frames, desc="Loading frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                pbar.update(1)

        cap.release()
        return frames, fps

    def save_video(self, frames: List[np.ndarray], output_path: Path, fps: float):
        """
        Save processed frames to video file.

        Args:
            frames: List of BGR frames to save
            output_path: Path for output video
            fps: Frames per second for output video
        """
        if not frames:
            logger.warning("No frames to save")
            return

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get frame dimensions
        height, width = frames[0].shape[:2]

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        logger.info(f"Saving video to: {output_path}")

        for frame in tqdm(frames, desc="Saving frames"):
            out.write(frame)

        out.release()
        logger.info("Video saved successfully")


class EmotionDetector:
    """Handles emotion detection using EmoNet model."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # Load models
        self.emonet = self._load_emonet()
        self.face_detector = SFDDetector(str(self.device))

        # Pre-allocate tensors for batch processing
        self.batch_tensor = torch.zeros(
            (config.batch_size, 3, config.image_size, config.image_size),
            device=self.device,
        )

    def _load_emonet(self) -> nn.Module:
        """Load and initialize EmoNet model."""
        model_path = Path(__file__).parent / f"emonet_{self.config.n_expression}.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading EmoNet model from: {model_path}")

        # Load state dict
        state_dict = torch.load(str(model_path), map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Initialize model
        model = EmoNet(n_expression=self.config.n_expression).to(self.device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model

    @torch.no_grad()
    def detect_faces_batch(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Detect faces in batch of frames.

        Args:
            frames: List of RGB frames

        Returns:
            List of detected face bounding boxes per frame
        """
        all_detections = []

        for frame in frames:
            # SFD detector expects BGR
            frame_bgr = frame[:, :, ::-1]
            detections = self.face_detector.detect_from_image(frame_bgr)

            # Limit to max faces per frame
            if len(detections) > self.config.max_faces_per_frame:
                detections = detections[: self.config.max_faces_per_frame]

            all_detections.append(detections)

        return all_detections

    @torch.no_grad()
    def process_batch(
        self, face_crops: List[np.ndarray]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Process a batch of face crops through EmoNet.

        Args:
            face_crops: List of RGB face crop images

        Returns:
            List of emotion predictions
        """
        if not face_crops:
            return []

        batch_size = len(face_crops)

        # Prepare batch tensor
        for i, face_crop in enumerate(face_crops):
            # Resize to model input size
            resized = cv2.resize(
                face_crop, (self.config.image_size, self.config.image_size)
            )

            # Convert to tensor and normalize
            self.batch_tensor[i] = (
                torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            )

        # Run inference
        batch_input = self.batch_tensor[:batch_size]
        outputs = self.emonet(batch_input)

        # Split outputs into individual predictions
        predictions = []
        for i in range(batch_size):
            pred = {
                key: value[i : i + 1] if isinstance(value, torch.Tensor) else value
                for key, value in outputs.items()
            }
            predictions.append(pred)

        return predictions


class Visualizer:
    """Handles visualization of emotion detection results."""

    def __init__(self, config: Config):
        self.config = config
        self.circumplex_path = Path(__file__).parent / "images/circumplex.png"

        # Cache circumplex image
        if self.circumplex_path.exists():
            self.circumplex_base = cv2.imread(str(self.circumplex_path))
        else:
            logger.warning(f"Circumplex image not found: {self.circumplex_path}")
            self.circumplex_base = None

    def create_valence_arousal_plot(
        self, valence: float, arousal: float, size: int = 512
    ) -> np.ndarray:
        """
        Create valence-arousal circumplex visualization.

        Args:
            valence: Valence value in range [-1, 1]
            arousal: Arousal value in range [-1, 1]
            size: Size of the output image

        Returns:
            BGR image with plotted point
        """
        if self.circumplex_base is None:
            # Create blank circumplex if image not available
            circumplex = np.ones((size, size, 3), dtype=np.uint8) * 255
        else:
            circumplex = cv2.resize(self.circumplex_base, (size, size))

        # Calculate position (arousal axis goes up, so invert)
        x = int((valence + 1.0) / 2.0 * size)
        y = int((1.0 - arousal) / 2.0 * size)

        # Draw position marker
        cv2.circle(circumplex, (x, y), 16, (0, 0, 255), -1)
        cv2.circle(circumplex, (x, y), 18, (0, 0, 0), 2)  # Border

        return circumplex

    def visualize_landmarks(
        self, face_crop: np.ndarray, heatmap: torch.Tensor
    ) -> np.ndarray:
        """
        Visualize facial landmarks on face crop.

        Args:
            face_crop: RGB face image
            heatmap: Landmark heatmap tensor

        Returns:
            RGB image with landmarks
        """
        # Resize heatmap to match face crop
        h, w = face_crop.shape[:2]
        heatmap_resized = nn.functional.interpolate(
            heatmap, size=(h, w), mode="bilinear", align_corners=False
        )

        result = face_crop.copy()

        # Draw each landmark
        for i in range(heatmap_resized.shape[1]):
            # Find peak position
            landmark_map = heatmap_resized[0, i]
            max_pos = (landmark_map == landmark_map.max()).nonzero(as_tuple=True)

            if len(max_pos[0]) > 0:
                y, x = int(max_pos[0][0]), int(max_pos[1][0])
                cv2.circle(result, (x, y), 4, (255, 255, 255), -1)
                cv2.circle(result, (x, y), 5, (0, 0, 0), 1)  # Border

        return result

    def create_frame_visualization(
        self,
        frame: np.ndarray,
        face_bbox: Optional[np.ndarray],
        face_crop: Optional[np.ndarray],
        prediction: Optional[Dict[str, torch.Tensor]],
    ) -> np.ndarray:
        """
        Create complete visualization for a single frame.

        Args:
            frame: Original RGB frame
            face_bbox: Face bounding box [x1, y1, x2, y2]
            face_crop: Cropped face image
            prediction: Emotion prediction results

        Returns:
            BGR visualization frame
        """
        h, w = frame.shape[:2]
        panel_size = h // 2

        # Create output canvas
        viz_width = w + panel_size
        viz = np.zeros((h, viz_width, 3), dtype=np.uint8)

        # Copy main frame (convert to BGR)
        viz[:, :w] = frame[:, :, ::-1]

        if face_bbox is not None and prediction is not None:
            # Draw face bounding box
            x1, y1, x2, y2 = face_bbox.astype(int)[:4]
            cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Add emotion label
            emotion_probs = nn.functional.softmax(prediction["expression"], dim=1)
            emotion_idx = torch.argmax(emotion_probs).item()
            emotion_label = self.config.emotion_classes[emotion_idx]  # type: ignore
            confidence = emotion_probs[0, emotion_idx].item()  # type: ignore

            label_text = f"{emotion_label} ({confidence:.2f})"
            label_pos = ((x1 + x2) // 2, y1 - 10)

            cv2.putText(
                viz,
                label_text,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # Add landmark visualization
            landmarks_viz = self.visualize_landmarks(face_crop, prediction["heatmap"])  # type: ignore
            landmarks_viz = cv2.resize(landmarks_viz, (panel_size, panel_size))
            viz[:panel_size, w:] = landmarks_viz[:, :, ::-1]

            # Add valence-arousal plot
            valence = prediction["valence"].clamp(-1, 1).item()
            arousal = prediction["arousal"].clamp(-1, 1).item()
            circumplex = self.create_valence_arousal_plot(valence, arousal, panel_size)
            viz[panel_size:, w:] = circumplex

            # Add V-A values as text
            va_text = f"V: {valence:.2f}, A: {arousal:.2f}"
            cv2.putText(
                viz,
                va_text,
                (w + 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return viz


@contextmanager
def timer(name: str):
    """Context manager for timing operations."""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} took {elapsed:.2f} seconds")


def process_video(video_path: Path, output_path: Path, config: Config) -> None:
    """
    Main processing pipeline for emotion detection in video.

    Args:
        video_path: Path to input video
        output_path: Path for output video
        config: Processing configuration
    """
    # Initialize components
    video_processor = VideoProcessor(config)
    detector = EmotionDetector(config)
    visualizer = Visualizer(config)

    # Load video
    with timer("Video loading"):
        frames, fps = video_processor.load_video(video_path)

    logger.info(f"Processing {len(frames)} frames in batches of {config.batch_size}")

    # Process frames in batches
    output_frames = []

    with timer("Emotion detection"):
        for batch_start in tqdm(
            range(0, len(frames), config.batch_size), desc="Processing batches"
        ):
            batch_end = min(batch_start + config.batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]

            # Detect faces in batch
            batch_detections = detector.detect_faces_batch(batch_frames)

            # Collect face crops and their metadata
            face_crops = []
            face_metadata = []  # (frame_idx, bbox)

            for i, (frame, detections) in enumerate(
                zip(batch_frames, batch_detections)
            ):
                if detections:
                    # Use first detected face
                    bbox = np.array(detections[0]).astype(int)
                    x1, y1, x2, y2 = bbox[:4]

                    # Extract face crop with bounds checking
                    y1, y2 = max(0, y1), min(frame.shape[0], y2)
                    x1, x2 = max(0, x1), min(frame.shape[1], x2)

                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        face_crops.append(face_crop)
                        face_metadata.append((i, bbox))

            # Process face crops through EmoNet
            if face_crops:
                predictions = detector.process_batch(face_crops)
            else:
                predictions = []

            # Create visualizations
            pred_idx = 0
            for i, frame in enumerate(batch_frames):
                if pred_idx < len(face_metadata) and face_metadata[pred_idx][0] == i:
                    # Frame has a detected face
                    _, bbox = face_metadata[pred_idx]
                    face_crop = face_crops[pred_idx]
                    prediction = predictions[pred_idx]
                    pred_idx += 1

                    viz_frame = visualizer.create_frame_visualization(
                        frame, bbox, face_crop, prediction
                    )
                else:
                    # No face detected
                    viz_frame = visualizer.create_frame_visualization(
                        frame, None, None, None
                    )

                output_frames.append(viz_frame)

    # Save output video
    with timer("Video saving"):
        video_processor.save_video(output_frames, output_path, fps)

    logger.info("Processing complete!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process video for emotion detection with batch processing"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.mp4",
        help="Path for output video file",
    )
    parser.add_argument(
        "--nclasses",
        type=int,
        default=8,
        choices=[5, 8],
        help="Number of emotion classes (5 or 8)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for GPU processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, etc.)",
    )

    args = parser.parse_args()

    # Configure
    config = Config(
        n_expression=args.nclasses, device=args.device, batch_size=args.batch_size
    )

    # Process video
    video_path = Path(args.video_path)
    output_path = Path(args.output_path)

    try:
        process_video(video_path, output_path, config)
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True

    main()
