import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import yt_dlp

from src.bouldering.media.video.video import Video


def download_youtube_video(
    url: str,
    out_dir: str,
    filename: str,
    resolution: Optional[int] = None,
    quiet: bool = False,
) -> str:
    """Download a YouTube video in MP4 format, optionally clipping by time and targeting a resolution.

    Args:
        url (str): Video URL.
        out_dir (str): Path to to output directory.
        filename (str): Output filename (e.g 'video.mp4').
        resolution (Optional[int], optional): Target resolution height in pixels. Defaults to None.
        quiet (bool, optional): Verbose if True. Defaults to False.

    Returns:
        str: The absolute path to the downloaded video.
    """
    out_path = os.path.join(out_dir, filename)

    if resolution is not None:
        fmt = (
            f"bestvideo[ext=mp4][vcodec*=avc1][height<={resolution}]"
            f"+bestaudio[ext=m4a]/"
            f"bestvideo[ext=mp4][height<={resolution}]+bestaudio/"
            f"best[height<={resolution}]"
        )
    else:
        fmt = (
            "bestvideo[ext=mp4][vcodec*=avc1]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio/best"
        )

    ydl_opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": out_path,
        "noplaylist": True,
        "quiet": quiet,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return out_path


def main():
    """Main script to run."""
    VIDEO_URL = "https://youtu.be/45KmZUc0CzA?si=v6acG49NSiif57bM"

    REPO_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = REPO_ROOT / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # temporary file
    tmp_dir = tempfile.mkdtemp(prefix="ydl_tmp_")

    # download the video
    try:
        with tempfile.TemporaryDirectory(prefix="ydl_full_") as tmp_dir:
            print(f"Downloading Youtube video {VIDEO_URL}")
            tmp_src_path = download_youtube_video(
                url=VIDEO_URL,
                out_dir=tmp_dir,
                filename="video.mp4",
                resolution=720,
                quiet=True,
            )
            print(f"Downloaded to {tmp_src_path}")

            # read, cut 0 - 7min
            print("reading tmp full video")
            video = Video.read(tmp_src_path)
            print("video read - OK")
            video_cut = video.cut(0.0, 7 * 60.0)
            print("video cut - OK")

            out_filename = str(DATA_DIR / "video.mp4")
            video_cut.write(out_filename)
            print(f"Video write - {out_filename}")
            print("Done")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
