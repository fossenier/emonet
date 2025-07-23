"""
Main entry point for processing the collected data in Parnian's userstudy

iSE Lab - Logan Fossenier - July 2025
"""

from collections import Counter
from config import Config
from emonet.models import load_emonet
from face_alignment.detection.sfd.sfd_detector import SFDDetector
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path
from process_face import FaceFrame, load_face_frames
from process_screen import ScreenFrame, load_screen_frames
from process_tags import TagFrame, load_tag_frames
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch


CONFIG = Config()


def main():
    # Create output directory
    os.makedirs("./userstudy_output", exist_ok=True)

    # Initialize EmoNet and face detector
    emonet: torch.nn.Module = load_emonet(
        CONFIG.pretrained_path, CONFIG.emotion_count, CONFIG.device
    )
    sfd_detector = SFDDetector(CONFIG.device)

    # Iterate through folders in input directory
    for folder_name in os.listdir("./userstudy_input/"):
        folder_path = Path("./userstudy_input/") / folder_name

        if folder_path.is_dir():
            # Create corresponding output folder
            output_folder = Path("./userstudy_output/") / folder_name
            output_folder.mkdir(exist_ok=True)

            # Get all files in the folder
            files = list(folder_path.glob("*"))

            screen, face, tag, gp3 = None, None, None, None
            found = 0
            # Identify files
            for file in files:
                filename = file.name
                print(f"Filename: {filename}")

                # gp3 tracker export
                if filename.endswith(".avi"):
                    found += 1
                    screen = file
                # TODO Parnian, if you record another format, just change this case
                # user recording
                elif filename.endswith(".mov"):
                    found += 1
                    face = file
                elif "tag_" in filename and filename.endswith(".csv"):
                    found += 1
                    tag = file
                elif "all_gaze" in filename and filename.endswith(".csv"):
                    found += 1
                    gp3 = file

            if not screen or not face or not tag or not gp3:
                print(
                    "Aborting, some directories in ./userstudy_input/ are missing files."
                )
                print(
                    f"{'screen' if screen else ''} {'face' if face else ''} {'tag' if tag else ''} {'gp3' if gp3 else ''}"
                )
                return
            if found != 4:
                print(
                    "Aborting, some directories in ./userstudy_input/ have too many files."
                )
                return

            # Assume face capture is of form
            # face_2025-07-22T11_02_44_6066.mov
            # Extract timestamps
            face_time = pd.to_datetime(
                str(face.name), format="face_%Y-%m-%dT%H_%M_%S_%f.mov"
            )

            # Get TIME column from gp3 csv
            gp3_df = pd.read_csv(gp3)
            time_col = [col for col in gp3_df.columns if "TIME" in col][0]
            gp3_time = pd.to_datetime(time_col, format="TIME(%Y/%m/%d %H:%M:%S.%f)")

            face_frames: List[FaceFrame] = load_face_frames(
                str(face.resolve()), face_time, emonet, sfd_detector
            )
            screen_frames: List[ScreenFrame] = load_screen_frames(
                str(screen.resolve()), gp3_time
            )
            tag_frames: List[TagFrame] = load_tag_frames(str(tag.resolve()))

            print("Frames processed!")
            print(
                f"Valence: {face_frames[3].valence} Arousal: {face_frames[3].arousal}"
            )

            print(f"Screen 100 timestamp: {screen_frames[100].timestamp}")
            print(f"Face 400 timestamp: {face_frames[400].timestamp}")

            #####
            # Now for the fun part, accumulating all the data into a video
            #####
            out_path = str(output_folder / (output_folder.name + "analysis.mp4"))
            process_video(screen_frames, face_frames, tag_frames, out_path)


def accumulate_interval_data(
    current_time, next_time, face_frames: List[FaceFrame], tag_frames: List[TagFrame]
) -> Tuple[List[FaceFrame], List[TagFrame]]:
    """Get all face/tag frames within the interval."""
    interval_faces = [f for f in face_frames if current_time <= f.timestamp < next_time]
    interval_tags = [t for t in tag_frames if current_time <= t.timestamp < next_time]
    return interval_faces, interval_tags


def aggregate_emotions(faces: List[FaceFrame]) -> Tuple[float, float, str]:
    """Average valence/arousal and get mode emotion."""
    if not faces:
        return 0.0, 0.0, "Unknown"

    valences = [f.valence for f in faces if f.valence is not None]
    arousals = [f.arousal for f in faces if f.arousal is not None]
    emotions = [f.emotion for f in faces]

    avg_valence = np.mean(valences) if valences else 0.0
    avg_arousal = np.mean(arousals) if arousals else 0.0
    mode_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "Unknown"

    return float(avg_valence), float(avg_arousal), mode_emotion


def create_graph_overlay(
    valence_history: List[float], width=300, height=100
) -> np.ndarray:
    """Create a simple graph as numpy array."""
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.plot(valence_history[-50:], "b-", linewidth=2)  # Last 50 points
    ax.set_ylim(-1, 1)
    ax.set_title("Valence", fontsize=10)
    ax.axis("off")

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    graph_array = np.asarray(buf)[:, :, :3]  # RGB only
    plt.close(fig)

    return graph_array


def process_video(
    screen_frames: List[ScreenFrame],
    face_frames: List[FaceFrame],
    tag_frames: List[TagFrame],
    output_path: str,
    fps: float = 10.0,
) -> None:
    """Main processing loop."""

    # Sort all frames by timestamp
    screen_frames.sort(key=lambda x: x.timestamp)
    face_frames.sort(key=lambda x: x.timestamp)
    tag_frames.sort(key=lambda x: x.timestamp)

    # Initialize video writer
    height, width = screen_frames[0].frame.shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    valence_history = []

    for i, screen_frame in enumerate(screen_frames):
        current_time = screen_frame.timestamp
        next_time = (
            screen_frames[i + 1].timestamp
            if i + 1 < len(screen_frames)
            else current_time
        )

        # Get interval data
        interval_faces, interval_tags = accumulate_interval_data(
            current_time, next_time, face_frames, tag_frames
        )

        # Aggregate emotions
        avg_valence, avg_arousal, mode_emotion = aggregate_emotions(interval_faces)
        valence_history.append(avg_valence)

        # Create visualization frame
        vis_frame = screen_frame.frame.copy()

        # Add emotion text
        cv2.putText(
            vis_frame,
            f"Emotion: {mode_emotion}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            vis_frame,
            f"V: {avg_valence:.2f} A: {avg_arousal:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Add visible tags
        visible_tags = [t.label for t in interval_tags if t.visible]
        if visible_tags:
            cv2.putText(
                vis_frame,
                f"Tags: {', '.join(visible_tags)}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        # Add graph overlay
        if len(valence_history) > 1:
            graph = create_graph_overlay(valence_history)
            graph_h, graph_w = graph.shape[:2]
            vis_frame[
                height - graph_h - 10 : height - 10, width - graph_w - 10 : width - 10
            ] = graph

        out.write(vis_frame)

    out.release()


if __name__ == "__main__":
    main()
