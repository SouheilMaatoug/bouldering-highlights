import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional


def run(cmd: List[str]) -> None:
    """Run a shell command.

    Args:
        cmd (List[str]): The command and its arguments.
    """
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def extract_audio_file(video_path: str, output_path: str, sr: Optional[int] = None) -> None:
    """Extract the audio track from a video into a WAV file.

    Args:
        video_path (str): The path to the video file.
        output_path (str): The path to write the audio file.
        sr (Optional[int], optional): The audio sample rate. Defaults to None.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
    ]
    if sr is not None:
        cmd += ["-ar", str(sr)]
    cmd += [output_path]
    run(cmd)
    return None


def merge_audio_video(output_path: str, video_path: str, audio_path: str) -> None:
    """Merge a video stream and an audio stream into a single file.

    Args:
        output_path (str): The output path.
        video_path (str): The path to the video file.
        audio_path (str): The path to the audio file.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        output_path,
    ]
    run(cmd)


def concat_videos(output_path: str, video_paths: list[str]) -> None:
    """Concatenate multiple videos using ffmpeg.

    Args:
        output_path (str): Output video path.
        video_paths (list[str]): List of video files to concatenate.
    """
    # Create ffmpeg concat list file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for vp in video_paths:
            f.write(f"file '{Path(vp).absolute()}'\n")
        list_path = f.name

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        output_path,
    ]

    subprocess.run(cmd, check=True)

    Path(list_path).unlink()
