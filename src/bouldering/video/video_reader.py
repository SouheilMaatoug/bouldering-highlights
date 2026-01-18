from pathlib import Path

import cv2

from src.bouldering.video.audio import Audio
from src.bouldering.video.sequence import Sequence


class VideoReader:
    """Class for read a video file from disk."""

    def __init__(self, filename: str) -> None:
        """Create a VideoReader instance.

        Args:
            filename (str): The path to the video file.
        """
        self.video_path = Path(filename)
        if not self.video_path.exists():
            raise FileNotFoundError(f"File not found {filename}.")

        self.cap = cv2.VideoCapture(filename)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video {filename}")

    def release(self) -> None:
        """Release the capture resource."""
        if getattr(self, "cap", None) is not None:
            self.cap.release()

    def __enter__(self) -> "VideoReader":
        """Open the context manager."""
        return self

    def __exit__(self, exec_type, exc, tb) -> None:
        """Close the context manager."""
        self.release()

    def extract_sequence(self) -> Sequence:
        """Extract the sequence of frames of the video.

        Returns:
            Sequence: The sequence of frames of the video.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        sequence = Sequence(frames, self.cap.get(cv2.CAP_PROP_FPS))
        return sequence

    def extract_audio(self) -> Audio:
        """Extract the audio track of the video.

        Returns:
            Audio: The audio track of the video.
        """
        audio = Audio.read(self.video_path)
        return audio
