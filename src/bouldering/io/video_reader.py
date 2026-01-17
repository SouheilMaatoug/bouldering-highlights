from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


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

    # metadata
    @property
    def fps(self) -> float:
        """Frame per second."""
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.frame_count / self.fps

    @property
    def width(self) -> int:
        """Frame width in pixels"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Frame height in pixels"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def metadata(self) -> dict:
        """Get the video metadata dictionary."""
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "resolution": (self.width, self.height),
            "path": str(self.video_path),
        }

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read the next frame.

        Returns:
            tuple[bool, Optional[np.ndarray]]: The boolean result and the frame if it exists.
        """
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        return True, frame

    def __iter__(self) -> Iterator:
        """Iterate on the video frames.

        Yields:
            frame (np.ndarray).
        """
        while True:
            ok, frame = self.read()
            if not ok:
                break
            yield frame

    def get_frame_idx(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get a frame at an index.

        Args:
            frame_idx (int): The index of the frame.

        Returns:
            np.ndarray: The frame if it exists.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.read()
        return frame if ret else None

    def get_frame_timestamp(self, timestamp: float) -> Optional[np.ndarray]:
        """Get a frame at a timestamp.

        Args:
            timestamp (float): The timestamp in seconds.

        Returns:
            Optional[np.ndarray]: The frame if it exists.
        """
        frame_idx = int(timestamp * self.fps)
        return self.get_frame_idx(frame_idx)

    def release(self) -> None:
        """Release the capture resource."""
        if getattr(self, "cap", None) is not None:
            self.cap.release()
