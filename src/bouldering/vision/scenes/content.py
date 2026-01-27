from typing import List

from scenedetect import open_video
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_manager import SceneManager

from src.bouldering.media.video.video import Video
from src.bouldering.vision.scenes.base import Scene


class ContentSceneSplitter:
    """
    Micro scene splitter using PySceneDetect (visual content changes).
    """

    def __init__(
        self,
        threshold: float = 27.0,
        min_scene_len_sec: float = 1.5,
        downscale: int = 2,
    ):
        self.threshold = threshold
        self.min_scene_len_sec = min_scene_len_sec
        self.downscale = downscale

    def split_range(self, video: Video, start_sec: float, end_sec: float) -> List[Scene]:
        """
        Split a time range of a video into visual scenes.

        Args:
            video (Video): Input video.
            start_sec (float): Start time in seconds.
            end_sec (float): End time in seconds.

        Returns:
            List[Scene]: Content-based sub-scenes.
        """
        svideo = open_video(str(video.sequence.video_path))

        manager = SceneManager()
        manager.add_detector(
            ContentDetector(
                threshold=self.threshold,
                min_scene_len=int(self.min_scene_len_sec * svideo.frame_rate),
            )
        )

        if self.downscale and self.downscale > 1:
            manager.downscale = self.downscale

        svideo.seek(FrameTimecode(start_sec, svideo.frame_rate))
        manager.detect_scenes(
            video=svideo,
            end_time=FrameTimecode(end_sec, svideo.frame_rate),
        )

        scenes = manager.get_scene_list()

        return [
            Scene(
                start_time=s.get_seconds(),
                end_time=e.get_seconds(),
                metadata={"source": "content"},
            )
            for s, e in scenes
        ]
