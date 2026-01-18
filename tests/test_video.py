from pathlib import Path

import pytest

from src.bouldering.media.audio.audio import Audio
from src.bouldering.media.video.io import VideoReader, VideoWriter
from src.bouldering.media.video.sequence import Sequence


def test_video_reader(make_audio_video_file, fps, size, n_frames, sample_rate, duration):
    """Test VideoReader on a temporary file."""
    filename = make_audio_video_file

    reader = VideoReader(str(filename))

    # extract sequence
    seq = reader.extract_sequence()
    assert seq.fps == fps
    assert seq.resolution == size
    assert seq.start == 0
    assert seq.n_frames == n_frames

    # extract audio
    audio = reader.extract_audio()
    assert audio.sample_rate == sample_rate
    assert pytest.approx(audio.duration, 0.5) == duration


def test_video_writer(tmp_path: Path, make_audio_file, make_video_file):
    """Test that VideoWriter writes a temporary file."""
    video_file = make_video_file
    audio_file = make_audio_file

    seq = Sequence.read(str(video_file))
    audio = Audio.read(str(audio_file))

    out = tmp_path / "out.mp4"
    writer = VideoWriter(str(out))
    writer.write(seq, audio)

    # file existence
    assert out.exists()
    assert out.stat().st_size > 0
