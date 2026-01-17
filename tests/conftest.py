from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def fps() -> float:
    """Frames per second."""
    return 10.0


@pytest.fixture(scope="session")
def n_frames() -> int:
    """Number of frames."""
    return 12


@pytest.fixture(scope="session")
def size() -> Tuple[int, int]:
    """Frame size."""
    return (64, 80)


@pytest.fixture(scope="session")
def codec() -> str:
    """codec."""
    return "mp4v"


@pytest.fixture(scope="function")
def make_frame(size):
    """Create a random frame."""
    w, h = size
    def _make():
        return np.random.randint(low=0, high=255, size=(h, w, 3), dtype=np.uint8)
    return _make

@pytest.fixture(scope="function")
def make_video(tmp_path: Path, codec, size, fps, n_frames, make_frame):
    """Create a synthetic video in a temporary file."""
    filename = "sample.mp4"
    output_path = tmp_path / filename
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size, True)
    for _ in range(n_frames):
        frame = make_frame()
        writer.write(frame)
    writer.release()
    return output_path
