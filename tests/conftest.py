import subprocess
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="session")
def fps() -> float:
    """Frames per second."""
    return 10.0


@pytest.fixture(scope="session")
def n_frames() -> int:
    """Number of frames."""
    return 50


@pytest.fixture(scope="session")
def duration(fps, n_frames) -> float:
    """Duration (s)."""
    return float(fps * n_frames)


@pytest.fixture(scope="session")
def sample_rate() -> int:
    """Sample rate (Hz)."""
    return int(16e3)


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
def make_video_file(tmp_path: Path, codec, size, fps, n_frames, make_frame):
    """Create a synthetic video in a temporary file."""
    filename = "video_sample.mp4"
    output_path = tmp_path / filename
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size, True)
    for _ in range(n_frames):
        frame = make_frame()
        writer.write(frame)
    writer.release()
    return output_path


@pytest.fixture(scope="function")
def make_audio(sample_rate, duration):
    """Generate a synthetic audio track."""
    n_samples = int(round(sample_rate * duration))
    data = (np.random.rand(n_samples) * 2) - 1
    return data


@pytest.fixture(scope="function")
def make_audio_file(tmp_path: Path, sample_rate, make_audio):
    """Create a synthetic audio in a temporary file."""
    filename = "audio_sample.wav"
    output_path = tmp_path / filename
    data = make_audio
    sf.write(str(output_path), data, sample_rate)
    return output_path


@pytest.fixture(scope="function")
def make_audio_video_file(tmp_path: Path, make_video_file, make_audio_file):
    """Create a synthetic video audio in a temporary file."""
    video_file = make_video_file
    audio_file = make_audio_file
    filename = "av_sample.mp4"
    output_path = tmp_path / filename
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_file),
        "-i",
        str(audio_file),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        str(output_path),
    ]
    _ = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path
