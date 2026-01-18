import math

from src.bouldering.media.audio.audio import Audio
from src.bouldering.media.video.io import VideoReader, VideoWriter
from src.bouldering.media.video.sequence import Sequence


class Video:
    """Class for Video instance."""

    def __init__(self, sequence: Sequence, audio: Audio) -> None:
        """Initialize the Video instance.

        Args:
            sequence (Sequence): The frames of the video.
            audio (Audio): The audio of the video
        """
        self.sequence = sequence
        self.audio = audio

    @classmethod
    def read(cls, path: str) -> "Video":
        """A class method to directly read a Video from a file.

        Args:
            path (str): The path to the video file.

        Returns:
            Video: A Video object.
        """
        vr = VideoReader(path)
        sequence = vr.extract_sequence()
        audio = vr.extract_audio()
        return cls(sequence, audio)

    @property
    def metadata(self) -> dict:
        """Get video metadata.

        Returns:
            dict: The metadata dictionary with keys 'video' and 'audio'.
        """
        return {
            "video": {
                "fps": self.sequence.fps,
                "n_frames": self.sequence.n_frames,
                "resolution": self.sequence.resolution,
            },
            "audio": {"sample_rate": self.audio.sample_rate, "duration": self.audio.duration},
        }

    def cut(self, start_time: float, end_time) -> "Video":
        """Cut the video between start time and end time (in seconds).

        Args:
            start_time (float): The start time in seconds.
            end_time (_type_): The end time in seconds.

        Returns:
            Video: A new Video object.
        """
        fps = float(self.sequence.fps)
        n_total = int(self.sequence.n_frames)

        start_frame = max(0, min(n_total, int(math.floor(start_time * fps))))
        end_frame = max(0, min(n_total, int(math.ceil(end_time * fps))))

        sequence_cut = self.sequence.cut(start_frame, end_frame)
        audio_cut = self.audio.cut(start_time, end_time)

        return Video(sequence_cut, audio_cut)

    def write(self, output_path: str) -> None:
        """Write a Video to a file.

        Args:
            output_path (str): The path to write the Video.
        """
        vw = VideoWriter(output_path)
        vw.write(self.sequence, self.audio)
