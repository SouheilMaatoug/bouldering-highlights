from pathlib import Path
from typing import Tuple

import cv2


class VideoWriter:
    """Class for writing a video."""

    def __init__(
        self, output_path: str, fps: float, resolution: Tuple[int, int], codec: str = "mp4v"
    ):
        """Initialize the VideoWriter instance.

        Args:
            output_path (str): The output path of the video.
            fps (float): Frames per second.
            resolution (Tuple[int, int]): Frame resolution in pixels.
            codec (str, optional): Writer codec. Defaults to "mp4v".
        """
        self.output_path = Path(output_path)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    def write(self, frame):
        """Write a single frame."""
        self.writer.write(frame)

    def close(self):
        """Close the writer."""
        self.writer.release()

    def __enter__(self) -> "VideoWriter":
        """Open the context manager."""
        return self

    def __exit__(self, exec_type, exc, tb) -> None:
        """Close the context manager."""
        self.close()
