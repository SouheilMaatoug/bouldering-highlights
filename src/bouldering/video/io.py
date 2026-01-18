from pathlib import Path

import cv2

from bouldering.video import ffmpeg
from src.bouldering import utils
from src.bouldering.video.audio import Audio
from src.bouldering.video.sequence import Sequence


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

    def release(self) -> None:
        """Release the capture resource."""
        if getattr(self, "cap", None) is not None:
            self.cap.release()

    def __enter__(self) -> "VideoReader":
        """Open the context manager."""
        return self

    def __exit__(self, exec_type, exc, tb) -> None:
        """Close the context manager."""
        self.release()

    def extract_sequence(self) -> Sequence:
        """Extract the sequence of frames of the video.

        Returns:
            Sequence: The sequence of frames of the video.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        sequence = Sequence(frames, self.cap.get(cv2.CAP_PROP_FPS))
        return sequence

    def extract_audio(self) -> Audio:
        """Extract the audio track of the video.

        Returns:
            Audio: The audio track of the video.
        """
        audio = Audio.read(self.video_path)
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
        fourcc = cv2.VideoWriter_fourcc(*codec)
        frame_writer = cv2.VideoWriter(output_path, fourcc, sequence.fps, sequence.resolution)
        for frame in sequence:
            frame_writer.write(frame)
        frame_writer.release()

    @staticmethod
    def write_audio(output_path: str, audio: Audio):
        """Write audio to the output file."""
        audio.write(output_path)

    def write(self, sequence: Sequence, audio: Audio):
        """Write a single frame."""

        # write the frames in a temporary file
        tmp_video = utils.create_tmp_file(suffix=".mp4")
        self.write_frames(tmp_video, self.codec, sequence)

        # write the audion in a temporary file
        tmp_audio = utils.create_tmp_file(suffix=".wav")
        self.write_audio(tmp_audio, audio)

        # merge audio video
        ffmpeg.merge_audio_video(self.output_path, tmp_video, tmp_audio)

        # clear tmp files
        utils.clear_file(tmp_video)
        utils.clear_file(tmp_audio)

    def close(self):
        """Close the writer."""
        self.writer.release()

    def __enter__(self) -> "VideoWriter":
        """Open the context manager."""
        return self

    def __exit__(self, exec_type, exc, tb) -> None:
        """Close the context manager."""
        self.close()
