from pathlib import Path

import soundfile as sf
from src.bouldering.media.audio.audio import Audio


def test_audio(tmp_path: Path, make_audio_file, duration, sample_rate):
    """Unit test for the Audio class."""
    audio_file = make_audio_file
    audio = Audio.read(str(audio_file))

    # metadata
    assert audio.sample_rate == sample_rate
    assert audio.duration == duration
    assert len(audio.samples) == sample_rate * duration

    # cut
    audio_cut = audio.cut(0.2, 2.3)
    assert audio_cut.sample_rate == sample_rate
    assert audio_cut.duration == 2.1

    # write
    out = tmp_path / "sample.wav"
    audio_cut.write(str(out))

    assert out.exists()
    assert out.stat().st_size > 0
    data, sr = sf.read(str(out))
    assert len(data) > 0
    assert sr == sample_rate
