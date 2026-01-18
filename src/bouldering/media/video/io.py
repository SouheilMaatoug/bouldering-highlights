from pathlib import Path

import cv2

from src.bouldering.media.audio.audio import Audio
from src.bouldering.media.video.sequence import Sequence
from src.bouldering.utils import ffmpeg, tmp


class VideoReader:
    """Class for read a video file from disk."""

    def __init__(self, filename: str) -> None:
        """Create a VideoReader instance.

        Args:
            filename (str): The path to the video file.
        """
        self.video_path = Path(filename)
        if not self.video_path.exists():
            raise FileNotFoundError(f"File not found {filename}.")

        self.cap = cv2.VideoCapture(filename)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video {filename}")

        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        """Frame per second."""
        return self._fps

    @property
    def width(self) -> int:
        """Frame width in pixels"""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels"""
        return self._height

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return self._nframes

    def extract_sequence(self) -> Sequence:
        """Extract the sequence of frames of the video.

        Returns:
            Sequence: The sequence of frames of the video.
        """
        return Sequence(
            video_path=self.video_path,
            resolution=(self.width, self.height),
            fps=self.fps,
            start=0,
            end=self.n_frames,
        )

    def extract_audio(self) -> Audio:
        """Extract the audio track of the video.

        Returns:
            Audio: The audio track of the video.
        """
        tmp_audio = tmp.create_tmp_file(suffix=".wav")
        ffmpeg.extract_audio_file(self.video_path, tmp_audio)
        audio = Audio.read(tmp_audio)
        tmp.clear_file(tmp_audio)

        return audio


class VideoWriter:
    """Class for writing a video."""

    def __init__(self, output_path: str) -> None:
        """Initialize the VideoWriter class."""
        self.output_path = Path(output_path)
        # frame writer
        self.codec = cv2.VideoWriter_fourcc(*"mp4v")

    @staticmethod
    def write_frames(output_path: str, codec: str, sequence: Sequence):
        """Write frames to the output file. without audio."""
        frame_writer = cv2.VideoWriter(output_path, codec, sequence.fps, sequence.resolution)
        for frame in sequence.frames():
            frame_writer.write(frame)
        frame_writer.release()

    @staticmethod
    def write_audio(output_path: str, audio: Audio):
        """Write audio to the output file."""
        audio.write(output_path)

    def write(self, sequence: Sequence, audio: Audio):
        """Write a single frame."""

        # write the frames in a temporary file
        tmp_video = tmp.create_tmp_file(suffix=".mp4")
        self.write_frames(tmp_video, self.codec, sequence)

        # write the audion in a temporary file
        tmp_audio = tmp.create_tmp_file(suffix=".wav")
        self.write_audio(tmp_audio, audio)

        # merge audio video
        ffmpeg.merge_audio_video(self.output_path, tmp_video, tmp_audio)

        # clear tmp files
        tmp.clear_file(tmp_video)
        tmp.clear_file(tmp_audio)
