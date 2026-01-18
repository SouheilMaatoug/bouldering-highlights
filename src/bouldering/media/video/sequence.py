# module for defining a sequence of frames

from typing import List, Tuple

import numpy as np


class Sequence:
    """Class for defining a sequence of frames."""

    def __init__(self, frames: List[np.ndarray], fps: float) -> None:
        """Initialize the Sequence class.

        Args:
            frames (List[np.ndarray]): A list of frames.
            fps (float): Frames per second.
        """
        self._frames = frames
        self._fps = fps

    @property
    def frames(self):
        return self._frames

    @property
    def fps(self):
        return self._fps

    @property
    def n_frames(self) -> int:
        """Get the number of frames.

        Returns:
            int: The number of frames.
        """
        return len(self.frames)

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get the resolution of the frames.

        Returns:
            Tuple[int, int]: Width, height.
        """
        h, w, _ = self.frames[0].shape
        return (w, h)

    def cut(self, start_frame: int, end_frame: int) -> "Sequence":
        """Cut a sequence between two frames.

        Args:
            start_frame (int): The start frame index.
            end_frame (int): The end frame index.

        Returns:
            Sequence: A new instance of Sequence.
        """
        return Sequence(self.frames[start_frame:end_frame], self.fps)
