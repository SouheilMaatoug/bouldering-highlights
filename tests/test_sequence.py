from src.bouldering.media.video.sequence import Sequence


def test_sequence(make_video_file, fps, size, n_frames):
    """Test the class Sequence."""
    video_file = make_video_file
    seq = Sequence.read(str(video_file))

    # metadata
    assert seq.fps == fps
    assert seq.resolution == size
    assert seq.start == 0
    assert seq.n_frames == n_frames

    # cut sequence
    seq_cut = seq.cut(20, 25)
    assert seq_cut.fps == seq.fps
    assert seq_cut.resolution == seq.resolution
    assert seq_cut.n_frames == 5
