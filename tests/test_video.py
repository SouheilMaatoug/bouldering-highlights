from pathlib import Path

import cv2
import numpy as np

from src.bouldering.io.video_reader import VideoReader
from src.bouldering.io.video_writer import VideoWriter


def test_video_reader(make_video, fps, size, n_frames):
    """Test that VideoReader reads a temporary file."""
    filename = make_video
    w, h = size
    reader = VideoReader(str(filename))


    # check metadata
    assert reader.fps == fps
    assert reader.frame_count == n_frames
    assert (reader.width, reader.height) == size

    # iterate over frames
    count = 0
    with VideoReader(str(filename)) as vr:
        for frame in vr:
            assert frame is not None
            assert frame.shape == (h, w, 3)
            count += 1
    assert count == n_frames


def test_video_writer(tmp_path: Path, codec, fps, size, n_frames):
    """Test that VideoWriter writes a temporary file."""
    out = tmp_path / "out.mp4"
    w, h = size

    with VideoWriter(output_path=str(out), codec=codec, fps=fps, resolution=size) as vw:
        for _ in range(n_frames):
            frame = np.random.randint(low=0, high=255, size=(h, w, 3), dtype=np.uint8)
            vw.write(frame)

    # file existence
    assert out.exists()
    assert out.stat().st_size > 0

    # verify we can read it
    cap = cv2.VideoCapture(str(out))
    _, frame = cap.read()
    cap.release()

    assert frame is not None
    assert frame.shape[0] == h
    assert frame.shape[1] == w
