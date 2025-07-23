"""
Takes the tag data output by the VS Code extension and then determines each
start / stop time for the tags that should be shown for each part of the video
and then provides the results as a list of TagFrames to be used in the final
combination.

iSE Lab Usask - Logan Fossenier - July 2025
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import csv
import pandas as pd


class TagFrame:
    def __init__(self, timestamp: pd.Timestamp, label: str, visible: bool) -> None:
        self.timestamp: pd.Timestamp = timestamp
        self.label: str = label
        self.visible: bool = visible


def load_tag_frames(tag_csv_path: str) -> List[TagFrame]:

    def validate_csv(filename: str) -> Tuple[bool, str]:
        # Track state for each tag
        tag_states = defaultdict(list)  # Will store list of visible values for each tag
        errors = []

        try:
            with open(filename, "r") as file:
                reader = csv.DictReader(file)

                # Check if required columns exist
                required_columns = {"tag", "timestamp", "visible"}
                if reader.fieldnames is None or not required_columns.issubset(
                    reader.fieldnames
                ):
                    missing = required_columns - set(reader.fieldnames or [])
                    return (False, f"Missing required columns: {missing}")

                rows = list(reader)

                if not rows:
                    return (False, "CSV file is empty")

                # Check first row
                first_row = rows[0]
                if (
                    first_row["tag"] != "start"
                    or first_row["visible"].lower() != "true"
                ):
                    errors.append("First row must have tag='start' and visible=True")

                # Check last row
                last_row = rows[-1]
                if last_row["tag"] != "start" or last_row["visible"].lower() != "false":
                    errors.append("Last row must have tag='start' and visible=False")

                # Process each row and track visible patterns per tag
                for i, row in enumerate(rows):
                    tag = row["tag"]
                    visible = row["visible"].lower() == "true"
                    tag_states[tag].append(visible)

                # Make sure each true has a falsefor each tag
                for tag, visible_list in tag_states.items():
                    if visible_list.count("true") != visible_list.count("false"):
                        errors.append(
                            f"Tag '{tag}' is invalid, unmatched visible true / false pattern."
                        )

                if errors:
                    return (
                        False,
                        "Validation failed:\n"
                        + "\n".join(f"- {error}" for error in errors),
                    )
                else:
                    return (True, "CSV validation passed! All rules satisfied.")

        except FileNotFoundError:
            return (False, f"File '{filename}' not found")
        except Exception as e:
            return (False, f"Error reading file: {e}")

    # Resulting data
    tag_frames: List[TagFrame] = []

    # Start by making sure the .csv is written properly
    proper, msg = validate_csv(tag_csv_path)
    if not proper:
        print(f"Error while loading tag frames: {msg}")
        return tag_frames

    # The count of current "visible"s to make sure that it stays visible until
    # the last "false"
    active: Dict[str, int] = {}

    e = ["tag", "timestamp", "visible"]
    # Start saving the frames
    with open(tag_csv_path, "r") as file:
        for row in csv.DictReader(file):
            # Grab data in the appropriate types
            tag = row["tag"]
            timestamp = pd.to_datetime(row["timestamp"])
            visible = True if row["visible"] == "True" else False

            # Make sure the tag is in the dict
            try:
                active[tag]
                # Great, it exists
            except:
                # Make the count zero
                active[tag] = 0

            # Skip 'start' (which includes the notion of stop)
            if tag == "start":
                continue

            # Track if the tag is active or inactive
            if visible:
                if active[tag] == 0:
                    # The tag is becoming visible from invisible
                    tag_frames.append(TagFrame(timestamp, tag, visible))
                active[tag] += 1
            else:
                if active[tag] == 1:
                    # The tag is becoming invisible from visible
                    tag_frames.append(TagFrame(timestamp, tag, visible))
                active[tag] -= 1

    return tag_frames
