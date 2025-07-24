"""
Main entry point for processing the collected data in Parnian's userstudy

iSE Lab - Logan Fossenier - July 2025
"""

from collections import Counter, deque
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
    print("Starting userstudy analysis")
    # Create output directory
    os.makedirs("./userstudy_output", exist_ok=True)

    # Initialize EmoNet and face detector onto the GPU
    emonet: torch.nn.Module = load_emonet(
        CONFIG.pretrained_path, CONFIG.emotion_count, CONFIG.device
    )
    sfd_detector = SFDDetector(CONFIG.device)

    # Output is exactly one .mp4, they will all appear in the same output folder
    # and get a unique name corresponding to their input folder
    output_folder = Path("./userstudy_output/")
    output_folder.mkdir(exist_ok=True)

    input_folder_list = os.listdir("./userstudy_input/")
    # Each folder in ./userstudy_input/ is considered a differnt test subject
    for i, folder_name in enumerate(input_folder_list):
        print(f"\nProcessing study {i} of {len(input_folder_list)}\n")
        folder_path = Path("./userstudy_input/") / folder_name

        if folder_path.is_dir():
            # Get all files in the folder
            files = list(folder_path.glob("*"))

            # Look for each needed file, make None to protect future loop iterations
            screen_file, face_file, tag_file, gp3_file = None, None, None, None
            found = 0
            # Identify files
            for file in files:
                filename = file.name
                # The GP3 Gazepoint software always exports .avi
                if filename.endswith(".avi"):
                    found += 1
                    screen_file = file
                # TODO Parnian, if you record another format, just change this case
                # The ffmpg user recording is always in .mov
                # TODO is it? it is on Mac at least...
                elif filename.endswith(".mov"):
                    found += 1
                    face_file = file
                # The VS Code extension must create the .csv to have tag_ in the name
                elif "tag_" in filename and filename.endswith(".csv"):
                    found += 1
                    tag_file = file
                # The GP3 Gazepoint software always exports all_gaze this way
                elif "all_gaze" in filename and filename.endswith(".csv"):
                    found += 1
                    gp3_file = file

            if not screen_file or not face_file or not tag_file or not gp3_file:
                print(
                    "Aborting, some directories in ./userstudy_input/ are missing files."
                )
                return
            if found != 4:
                print(
                    "Aborting, some directories in ./userstudy_input/ have too many files."
                )
                print(
                    "*or perhaps some names are triggering two search cases, look in the code*"
                )
                return

            # To synchronize the videos, determine the precise moment each one starts

            # The VS Code extension will make ffmpeg store the date formatted in the filename
            # face_2025-07-22T11_02_44_6066.mov
            face_time = pd.to_datetime(
                str(face_file.name), format="face_%Y-%m-%dT%H_%M_%S_%f.mov"
            )

            # GP3's software will handily give a precise start time in the all_gaze .csv file
            gp3_df = pd.read_csv(gp3_file)
            time_col = [col for col in gp3_df.columns if "TIME" in col][0]
            gp3_time = pd.to_datetime(time_col, format="TIME(%Y/%m/%d %H:%M:%S.%f)")

            # Parse the files for their data, these lists can be iterated chronologically
            face_frames: List[FaceFrame] = load_face_frames(
                str(face_file.resolve()), face_time, emonet, sfd_detector
            )
            screen_frames: List[ScreenFrame] = load_screen_frames(
                str(screen_file.resolve()), gp3_time
            )
            tag_frames: List[TagFrame] = load_tag_frames(str(tag_file.resolve()))

            #####
            # Now for the fun part, accumulating all the data into a video
            #####
            out_path = str(output_folder / (output_folder.name + "analysis.mp4"))
            process_video(screen_frames, face_frames, tag_frames, out_path)

    print(f"###\nProcessed all {len(input_folder_list)} studies\n###")


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


def create_sliding_plot(
    data_window, title="", color_positive="green", color_negative="red"
):
    """Create a styled matplotlib plot for valence/arousal sliding window with black background"""
    # Higher resolution for better line quality
    fig, ax = plt.subplots(figsize=(4, 1.5), dpi=150)

    # Set up the plot
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, len(data_window) - 1)

    # Remove x-axis labels
    ax.set_xticks([])

    # Style the plot with black background
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    # Add grid with light color
    ax.grid(True, alpha=0.3, linestyle="--", color="white")

    # Add zero line in white
    ax.axhline(y=0, color="white", linewidth=0.8, alpha=0.5)

    # Get data
    x = np.arange(len(data_window))
    y = np.array(data_window)

    # Plot the line in white
    ax.plot(x, y, color="white", linewidth=1.5, zorder=3)

    # Fill area under curve with proper color separation
    # For positive values (above 0)
    ax.fill_between(
        x, y, 0, where=(y >= 0), color=color_positive, alpha=0.8, interpolate=True  # type: ignore
    )
    # For negative values (below 0)
    ax.fill_between(
        x, y, 0, where=(y <= 0), color=color_negative, alpha=0.8, interpolate=True  # type: ignore
    )

    # Style y-axis label and ticks in white
    ax.set_ylabel("", fontsize=8)
    ax.tick_params(axis="y", colors="white")

    # Style title in white
    ax.set_title(title, fontsize=12, pad=10, color="white")

    # Set spines (borders) to white
    for spine in ax.spines.values():
        spine.set_color("white")

    # Tight layout to prevent title cutoff
    fig.tight_layout()

    # Convert to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    plot_array = np.asarray(buf)[:, :, :3]  # RGB only
    plt.close(fig)

    return plot_array


def create_info_panel(tags, emotion, width, height, valence, arousal):
    """Create an info panel with emotion, valence/arousal, and flagged moments

    Args:
        tags: List of tags (used as flagged moments)
        emotion: Predicted emotion string
        width: Panel width in pixels
        height: Panel height in pixels
        valence: Float value for valence
        arousal: Float value for arousal
    """
    import numpy as np
    import cv2

    # Create black background
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate box dimensions
    box_height = height // 3

    # Calculate font scale based on box dimensions
    # Adjust these multipliers to fine-tune text size
    base_font_scale = min(width / 400, box_height / 60)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(base_font_scale * 2))

    # Helper function to get text color based on value
    def get_value_color(value):
        if value > 0:
            return (0, 255, 0)  # Green for positive
        elif value < 0:
            return (0, 0, 255)  # Red for negative
        else:
            return (255, 255, 255)  # White for zero

    # Box 1: Predicted emotion
    y_center = box_height // 2
    text = f"Predicted emotion: {emotion}"
    text_size = cv2.getTextSize(text, font, base_font_scale, thickness)[0]
    text_y = y_center + text_size[1] // 2
    cv2.putText(
        panel, text, (20, text_y), font, base_font_scale, (255, 255, 255), thickness
    )

    # Box 2: Valence and Arousal (split into two columns)
    box_width = width // 2

    # Calculate vertical center for both valence and arousal
    valence_text = f"Valence: {valence:.2f}"
    arousal_text = f"Arousal: {arousal:.2f}"

    # Get text heights to center vertically
    text_size_v = cv2.getTextSize(valence_text, font, base_font_scale, thickness)[0]
    text_size_a = cv2.getTextSize(arousal_text, font, base_font_scale, thickness)[0]

    # Center position for second box
    y_center = box_height + box_height // 2 + text_size_v[1] // 2

    # Valence (left half)
    text_parts = valence_text.split(": ")

    # Draw "Valence: " in white
    cv2.putText(
        panel,
        text_parts[0] + ": ",
        (20, y_center),
        font,
        base_font_scale,
        (255, 255, 255),
        thickness,
    )

    # Get width of "Valence: " to position the number
    label_size = cv2.getTextSize(
        text_parts[0] + ": ", font, base_font_scale, thickness
    )[0]

    # Draw the value in appropriate color
    cv2.putText(
        panel,
        text_parts[1],
        (20 + label_size[0], y_center),
        font,
        base_font_scale,
        get_value_color(valence),
        thickness,
    )

    # Arousal (right half)
    text_parts = arousal_text.split(": ")

    # Draw "Arousal: " in white
    cv2.putText(
        panel,
        text_parts[0] + ": ",
        (box_width + 20, y_center),
        font,
        base_font_scale,
        (255, 255, 255),
        thickness,
    )

    # Get width of "Arousal: " to position the number
    label_size = cv2.getTextSize(
        text_parts[0] + ": ", font, base_font_scale, thickness
    )[0]

    # Draw the value in appropriate color
    cv2.putText(
        panel,
        text_parts[1],
        (box_width + 20 + label_size[0], y_center),
        font,
        base_font_scale,
        get_value_color(arousal),
        thickness,
    )

    # Box 3: Flagged moments
    y_center = 2 * box_height + box_height // 2
    if tags:
        # Draw "Flagged moments: " in red
        label_text = "Flags: "
        label_size = cv2.getTextSize(label_text, font, base_font_scale, thickness)[0]

        # Join tags with comma separation
        tags_text = ", ".join(tags)

        # Check if full text needs wrapping
        full_text = label_text + tags_text
        text_size = cv2.getTextSize(full_text, font, base_font_scale, thickness)[0]

        if text_size[0] > width - 40:
            # Simple wrapping for long text
            words = tags_text.split(", ")
            lines = [(label_text, True)]  # First line with label (True = is label)
            current_line = ""

            for word in words:
                test_line = current_line + ", " + word if current_line else word
                test_size = cv2.getTextSize(
                    label_text + test_line if not lines[0][0] else test_line,
                    font,
                    base_font_scale,
                    thickness,
                )[0]

                if test_size[0] <= width - 40:
                    current_line = test_line
                else:
                    if current_line:
                        if len(lines) == 1:
                            lines[0] = (label_text + current_line, False)
                        else:
                            lines.append((current_line, False))
                    current_line = word

            if current_line:
                if len(lines) == 1:
                    lines[0] = (label_text + current_line, False)
                else:
                    lines.append((current_line, False))

            # Draw wrapped text centered in the box
            total_height = len(lines) * text_size[1]
            start_y = 2 * box_height + (box_height - total_height) // 2 + text_size[1]

            for i, (line, is_label) in enumerate(lines):
                y_pos = start_y + i * int(text_size[1] * 1.2)
                if i == 0:
                    # First line - draw label in red and rest in white
                    cv2.putText(
                        panel,
                        label_text,
                        (20, y_pos),
                        font,
                        base_font_scale,
                        (0, 0, 255),
                        thickness,
                    )
                    rest_text = line[len(label_text) :]
                    cv2.putText(
                        panel,
                        rest_text,
                        (20 + label_size[0], y_pos),
                        font,
                        base_font_scale,
                        (255, 255, 255),
                        thickness,
                    )
                else:
                    cv2.putText(
                        panel,
                        line,
                        (20, y_pos),
                        font,
                        base_font_scale,
                        (255, 255, 255),
                        thickness,
                    )
        else:
            # Draw single line centered
            text_y = y_center + text_size[1] // 2
            cv2.putText(
                panel,
                label_text,
                (20, text_y),
                font,
                base_font_scale,
                (0, 0, 255),
                thickness,
            )
            cv2.putText(
                panel,
                tags_text,
                (20 + label_size[0], text_y),
                font,
                base_font_scale,
                (255, 255, 255),
                thickness,
            )
    # If no tags, nothing is displayed

    return panel


# Replace the process_video function with this updated version
def process_video(
    screen_frames: List[ScreenFrame],
    face_frames: List[FaceFrame],
    tag_frames: List[TagFrame],
    output_path: str,
    fps: float = 10.0,
) -> None:
    """Main processing loop with sidebar visualization."""

    # Sort all frames by timestamp
    screen_frames.sort(key=lambda x: x.timestamp)
    face_frames.sort(key=lambda x: x.timestamp)
    tag_frames.sort(key=lambda x: x.timestamp)

    # Get original dimensions
    orig_height, orig_width = screen_frames[0].frame.shape[:2]

    # Calculate new dimensions
    # Main video on left, sidebar on right (half the height of original as width)
    sidebar_width = orig_height // 2
    total_width = orig_width + sidebar_width

    # Initialize video writer with new dimensions
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, orig_height))

    # Initialize sliding windows
    window_size = 50
    valence_window = deque(maxlen=window_size)
    arousal_window = deque(maxlen=window_size)

    # Initialize with zeros
    for _ in range(window_size):
        valence_window.append(0.0)
        arousal_window.append(0.0)

    visible_tags = set()

    # Load the image once before the loop
    try:
        placeholder_img = cv2.imread(
            "./images/circumplex.png"
        )  # Replace with your image path
        if placeholder_img is None:
            # Create a placeholder if image not found
            placeholder_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Resize to square size
        square_size = sidebar_width
        placeholder_img = cv2.resize(placeholder_img, (square_size, square_size))
    except:
        # Fallback to gray square if image loading fails
        square_size = sidebar_width
        placeholder_img = np.ones((square_size, square_size, 3), dtype=np.uint8) * 128

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

        # Update sliding windows
        valence_window.append(avg_valence)
        arousal_window.append(avg_arousal)

        # Create the main frame
        main_frame = np.zeros((orig_height, total_width, 3), dtype=np.uint8)

        # Place screen recording on the left
        main_frame[:, :orig_width] = screen_frame.frame

        # Create sidebar components
        sidebar_height = orig_height
        square_size = sidebar_width  # Since sidebar_width = orig_height // 2

        # Upper square - create emotion plot on top of the image
        upper_square = placeholder_img.copy()

        # Draw the emotion circle
        # Define the invisible circle space (7/8 of the way to edge)
        center_x = square_size // 2
        center_y = square_size // 2
        max_radius = (square_size // 2) * 7 // 8

        # Map valence (-1 to 1) to x position
        # valence -1 = left edge of circle, valence 1 = right edge of circle
        circle_x = int(center_x + (avg_valence * max_radius))

        # Map arousal (-1 to 1) to y position
        # arousal 1 = top of circle (lower y value), arousal -1 = bottom of circle (higher y value)
        circle_y = int(center_y - (avg_arousal * max_radius))

        # Draw the red circle
        cv2.circle(
            upper_square, (circle_x, circle_y), 8, (0, 0, 255), -1
        )  # Red filled circle
        cv2.circle(
            upper_square, (circle_x, circle_y), 10, (0, 0, 0), 2
        )  # Black outline

        # Lower square divided into three horizontal sections
        section_height = square_size // 3

        # Create valence plot
        valence_plot = create_sliding_plot(valence_window, "Valence")
        valence_plot_resized = cv2.resize(valence_plot, (square_size, section_height))

        # Create arousal plot (using same green/red colors as valence)
        arousal_plot = create_sliding_plot(
            arousal_window, "Arousal", color_positive="green", color_negative="red"
        )
        arousal_plot_resized = cv2.resize(arousal_plot, (square_size, section_height))

        # Update visible tags
        for t in interval_tags:
            if t.visible:
                visible_tags.add(t.label)
            else:
                visible_tags.discard(t.label)

        # Create info panel
        info_panel = create_info_panel(
            list(visible_tags),
            mode_emotion,
            square_size,
            section_height,
            avg_valence,
            avg_arousal,
        )

        # Assemble sidebar
        sidebar_x = orig_width

        # Upper square
        main_frame[:square_size, sidebar_x:] = upper_square

        # Lower square - three sections
        y_offset = square_size
        main_frame[y_offset : y_offset + section_height, sidebar_x:] = (
            valence_plot_resized
        )
        y_offset += section_height
        main_frame[y_offset : y_offset + section_height, sidebar_x:] = (
            arousal_plot_resized
        )
        y_offset += section_height
        main_frame[y_offset : y_offset + section_height, sidebar_x:] = info_panel

        out.write(main_frame)

        if i % 60 == 0:
            print(f"Assembled {i} analysis output frames")

    out.release()


if __name__ == "__main__":
    main()
