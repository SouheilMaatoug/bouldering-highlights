import math
from typing import List

import numpy as np

from src.bouldering.media.audio.audio import Audio
from src.bouldering.media.video.io import VideoReader, VideoWriter
from src.bouldering.media.video.sequence import Sequence
from src.bouldering.utils import ffmpeg, tmp


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

    @classmethod
    def concatenate(cls, videos: List["Video"]) -> "Video":
        """Concatenate multiple Video objects into a single Video.

        This method materializes intermediate files and performs
        video concatenation using ffmpeg.

        Args:
            videos (List[Video]): Videos to concatenate.

        Returns:
            Video: Concatenated video.
        """
        if not videos:
            raise ValueError("No videos to concatenate")

        fps = videos[0].sequence.fps
        resolution = videos[0].sequence.resolution
        sample_rate = videos[0].audio.sample_rate

        # ------------------------------------------
        # 1. Validate consistency
        # ------------------------------------------
        for v in videos:
            if v.sequence.fps != fps:
                raise ValueError("All videos must have same FPS")
            if v.sequence.resolution != resolution:
                raise ValueError("All videos must have same resolution")
            if v.audio.sample_rate != sample_rate:
                raise ValueError("All audios must have same sample rate")

        # ------------------------------------------
        # 2. Write each video part to temp video
        # ------------------------------------------
        tmp_videos = []

        for v in videos:
            tmp_video = tmp.create_tmp_file(suffix=".mp4")
            writer = VideoWriter(tmp_video)
            writer.write(v.sequence, v.audio)
            tmp_videos.append(tmp_video)

        # ------------------------------------------
        # 3. Concatenate video files (ffmpeg)
        # ------------------------------------------
        merged_video_path = tmp.create_tmp_file(suffix=".mp4")
        ffmpeg.concat_videos(merged_video_path, tmp_videos)

        # ------------------------------------------
        # 4. Concatenate audio in memory
        # ------------------------------------------
        audio_samples = [v.audio.samples for v in videos]
        merged_audio = Audio(
            samples=np.concatenate(audio_samples, axis=0),
            sample_rate=sample_rate,
        )

        # ------------------------------------------
        # 5. Cleanup temp video parts
        # ------------------------------------------
        for p in tmp_videos:
            tmp.clear_file(p)

        # ------------------------------------------
        # 6. Create final Sequence
        # ------------------------------------------
        total_frames = sum(v.sequence.n_frames for v in videos)

        merged_sequence = Sequence(
            video_path=merged_video_path,
            fps=fps,
            resolution=resolution,
            start=0,
            end=total_frames,
        )

        return cls(merged_sequence, merged_audio)
