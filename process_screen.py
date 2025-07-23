"""
Takes the user screen capture video, timestamps it, and then provides the results
as a list of ScreenFrames to be used in the final combination.

iSE Lab Usask - Logan Fossenier - July 2025
"""

from typing import List

import cv2
import cv2.typing as cvt
import pandas as pd


class ScreenFrame:
    def __init__(self, timestamp: pd.Timestamp, frame: cvt.MatLike) -> None:
        self.timestamp: pd.Timestamp = timestamp
        self.frame: cvt.MatLike = frame


def load_screen_frames(
    screen_video_path: str,
    start_time: pd.Timestamp,
) -> List[ScreenFrame]:
    print(f"Loading ScreenFrames for video {screen_video_path}")

    # Resulting data
    screen_frames: List[ScreenFrame] = []

    video_capture = cv2.VideoCapture(screen_video_path)
    # Grab the fps for data synchronization
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    i = 0
    while video_capture.isOpened():
        # Increment the time for each frame to maintain synchronization
        current_time = start_time + pd.Timedelta(milliseconds=i * (1000 / fps))
        ret, frame = video_capture.read()
        if not ret:
            break
        screen_frames.append(ScreenFrame(current_time, frame))
        if i % 60 == 0:
            print(f"Processed {i} screen video frames")
        i += 1

    return screen_frames
