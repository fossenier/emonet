"""
Takes the user study face capture video, uses the EmoNet prediction model, and
then provides the results as a list of FaceFrames to be used in the final
combination.

iSE Lab Usask - Logan Fossenier - July 2025
"""

from config import Config
from face_alignment.detection.sfd.sfd_detector import SFDDetector
from typing import Dict, List

import cv2
import cv2.typing as cvt
import numpy as np
import pandas as pd
import torch

CONFIG = Config()


class FaceFrame:
    def __init__(
        self,
        timestamp: pd.Timestamp,
        valence: float | None,
        arousal: float | None,
        emotion: str,
    ) -> None:
        self.timestamp: pd.Timestamp = timestamp
        self.valence: float | None = valence
        self.arousal: float | None = arousal
        self.emotion: str = emotion


def load_face_frames(
    face_video_path: str,
    start_time: pd.Timestamp,
    net: torch.nn.Module,
    sfd_detector: SFDDetector,
) -> List[FaceFrame]:
    print(f"Loading FaceFrames for video {face_video_path}")

    # Resulting data
    face_frames: List[FaceFrame] = []
    # Load video RGB frames for EmoNet
    rgb_frames: List[cvt.MatLike] = []

    video_capture = cv2.VideoCapture(face_video_path)
    # Grab the fps for data synchronization
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        # Necessary conversion for model
        rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Process the video frame by frame
    for i, frame in enumerate(rgb_frames):
        # Increment the time for each frame to maintain synchronization
        current_time = start_time + pd.Timedelta(milliseconds=i * (1000 / fps))
        # Reset / assume there will be no prediction, so that previous runs'
        # predictions don't get erroneously saved while frames are iterated
        emotion: str = ""
        valence: float | None = None
        arousal: float | None = None

        # Run face detector:
        with torch.no_grad():
            # The slicing flips the image to BGR from RGB
            detected_faces = sfd_detector.detect_from_image(frame)

        # EmoNet can be run when there are faces
        if len(detected_faces) > 0:
            # Just do the first face
            bbox = np.array(detected_faces[0]).astype(np.int32)
            face_crop = frame[bbox[1] : bbox[3], bbox[0] : bbox[2], :]

            # Resize frame
            resized = cv2.resize(face_crop, (CONFIG.image_size, CONFIG.image_size))
            # Turn frame [height, width, channels] into PyTorch [channels, height, width]
            # and normalize as [0.0, 1.0]
            frame_tensor = (
                torch.Tensor(resized).permute(2, 0, 1).to(CONFIG.device) / 255.0
            )

            with torch.no_grad():
                result: Dict[str, torch.Tensor] = net(frame_tensor.unsqueeze(0))

            emotion_idx = (
                torch.argmax(torch.nn.functional.softmax(result["expression"], dim=1))
                .cpu()
                .item()
            )
            valence = float(result["valence"].clamp(-1.0, 1.0))
            arousal = float(result["arousal"].clamp(-1.0, 1.0))
            emotion = CONFIG.emotion_classes[int(emotion_idx)]

        # Accumulate each processed frame
        face_frames.append(FaceFrame(current_time, valence, arousal, emotion))

        if i % 120 == 0:
            print(f"Processed {i * 120} face video frames.")
            print(f"Valence: {valence} Arousal: {arousal}")

    return face_frames
