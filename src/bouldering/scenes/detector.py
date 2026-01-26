from typing import List, Tuple

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode


class SceneDetector:
    """
    Scene detector based on visual content changes using PySceneDetect.
    """

    def __init__(
        self,
        threshold: float = 27.0,
        min_scene_len_sec: float = 1.5,
        downscale: int = 2,
    ) -> None:
        """
        Initialize the scene detector.

        Args:
            threshold (float): ContentDetector threshold.
            min_scene_len_sec (float): Minimum scene duration in seconds.
            downscale (int): Downscale factor for faster processing.
        """
        self.threshold = threshold
        self.min_scene_len_sec = min_scene_len_sec
        self.downscale = downscale

    def detect(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
    ) -> List[Tuple[float, float]]:
        """
        Detect scenes within a time range.

        Args:
            video_path (str): Path to the video file.
            start_sec (float): Start time in seconds.
            end_sec (float): End time in seconds.

        Returns:
            List[Tuple[float, float]]: List of (start_sec, end_sec) scenes.
        """
        video = open_video(video_path)
        scene_manager = SceneManager()

        scene_manager.add_detector(
            ContentDetector(
                threshold=self.threshold,
                min_scene_len=int(self.min_scene_len_sec * video.frame_rate),
            )
        )

        if self.downscale and self.downscale > 1:
            scene_manager.downscale = self.downscale

        video.seek(FrameTimecode(start_sec, video.frame_rate))

        scene_manager.detect_scenes(
            video=video,
            end_time=FrameTimecode(end_sec, video.frame_rate),
        )

        scenes = scene_manager.get_scene_list()

        return [(start.get_seconds(), end.get_seconds()) for start, end in scenes]
