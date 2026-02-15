# module for defining a sequence of frames

from typing import Iterator, Tuple

import cv2
import numpy as np


class Sequence:
    """Class for defining a (lazy) sequence of frames."""

    def __init__(
        self,
        video_path: str,
        fps: float,
        resolution: Tuple[int, int],
        start: int = 0,
        end: int = 1,
    ) -> None:
        """Initialize the Sequence class.

        Args:
            video_path (str): The path to the video file containing the sequence of frames.
            fps (float): Frames per second.
            resolution (Tuple[int, int]): Width, height.
            start (int, optional): Start frame. Defaults to 0.
            end (Optional[int], optional): End frame. Defaults to 1.
        """
        self.video_path = video_path
        self._resolution = resolution
        self._fps = fps
        self.start = start
        self.end = end

    @classmethod
    def read(cls, path: str) -> "Sequence":
        from src.bouldering.media.video.io import VideoReader

        vr = VideoReader(path)
        return vr.extract_sequence()

    @property
    def n_frames(self) -> int:
        """Get the number of frames.

        Returns:
            int: The number of frames.
        """
        return self.end - self.start

    @property
    def fps(self):
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get the resolution of the frames.

        Returns:
            Tuple[int, int]: Width, height.
        """
        return self._resolution

    def frame(self, idx: int) -> np.ndarray:
        """Extract a particular frame by its index.

        Args:
            idx (int): The index of the frame.

        Returns:
            np.ndarray: The frame at the index.
        """
        frame_idx = idx + self.start
        if frame_idx >= self.end:
            raise ValueError(f"index out of bounds! Maximum frames = {self.n_frames}")

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx + self.start)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame at index {idx}")
        cap.release()
        return frame

    def frames(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)

        current = self.start
        while True:
            if self.end is not None and current >= self.end:
                break

            ret, frame = cap.read()
            if not ret:
                break

            yield frame
            current += 1

        cap.release()

    def cut(self, start_frame: int, end_frame: int) -> "Sequence":
        """Cut a sequence between two frames.

        Args:
            start_frame (int): The start frame index.
            end_frame (int): The end frame index.

        Returns:
            Sequence: A new instance of Sequence.
        """
        return Sequence(
            self.video_path,
            self.fps,
            self.resolution,
            self.start + start_frame,
            self.start + end_frame,
        )
