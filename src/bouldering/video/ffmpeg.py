import subprocess


def run(cmd: list):
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def extract_audio_file(video_path: str, output_path: str, sr: int = 44100):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "2", "-ar", str(sr), output_path]
    run(cmd)
    return None


def merge_audio_video(output_path: str, video_path: str, audio_path: str):
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
