from src.bouldering.media.audio.audio import Audio
from src.bouldering.media.video.io import VideoReader, VideoWriter
from src.bouldering.media.video.sequence import Sequence


class Video:
    """Class for Video instance."""

    def __init__(self, sequence: Sequence, audio: Audio):
        self.sequence = sequence
        self.audio = audio

    @classmethod
    def read(cls, path: str) -> "Video":
        vr = VideoReader(path)
        sequence = vr.extract_sequence()
        audio = vr.extract_audio()
        return cls(sequence, audio)

    @property
    def metadata(self) -> dict:
        return {
            "video": {
                "fps": self.sequence.fps,
                "n_frames": self.sequence.n_frames,
                "resolution": self.sequence.resolution,
            },
            "audio": {"sample_rate": self.audio.sample_rate, "duration": self.audio.duration},
        }

    @staticmethod
    def time2frame(time: float, fps: float):
        """Convert timestamp to frame number."""
        return int(round(time * fps))

    def cut(self, start_time: float, end_time) -> "Video":
        fps = self.metadata["video"]["fps"]
        start_frame = self.time2frame(start_time, fps)
        end_frame = self.time2frame(end_time, fps)
        sequence_cut = self.sequence.cut(start_frame, end_frame)
        audio_cut = self.audio.cut(start_time, end_time)
        return Video(sequence_cut, audio_cut)

    def write(self, output_path: str) -> None:
        vw = VideoWriter(output_path)
        vw.write(self.sequence, self.audio)
