"""
Synchronizes a GP3 Gaze screen capture with an ffmpeg created face recording.
Passes the frames of the face to the emonet valance / arousal / emotion prediction
model and also accepts .csv data that includes landmark moments from a user
study. This all combines into a nice overview visualization of the user's on screen
interactions, with emotional analysis.

iSE Lab Usask - Logan Fossenier - July 2025
"""

from typing import List, Tuple

import cv2
import cv2.typing as cvt
import pandas as pd
import time


class CaptureFrame:
    def __init__(self, timestamp: int, frame: cvt.MatLike) -> None:
        """
        Args:
            timestamp (int): The timestamp at which the frame occurs.
            frame (cvt.MatLike): The frame data, expected to be compatible with cvt.MatLike.
        """
        self.timestamp = timestamp
        self.frame = frame





def main() -> None:
    # This stream is for the frames in the screen capture via GP3 (should be 10 fps)
    capture_stream: List[Tuple[int, cvt.MatLike]] = []

    face_stream: List[
        Tuple[
            int,
            cvt.MatLike,
        ]
    ]

    dt = pd.to_datetime("my_str")
    ms = int(dt.timestamp() * 1000)


if __name__ == "__main__":
    main()
