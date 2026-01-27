from typing import List

from src.bouldering.media.video.video import Video
from src.bouldering.vision.ocr.splitter import SceneSplitterOCR
from src.bouldering.vision.scenes.base import Scene, SceneSplitter


class OCRSceneSplitter(SceneSplitter):
    """
    Macro scene splitter based on OCR overlays (e.g. 'BOULDER N').
    """

    def __init__(self, splitter: SceneSplitterOCR):
        """
        Args:
            splitter (SceneSplitterOCR): OCR-based splitter logic.
        """
        self._splitter = splitter

    def split(self, video: Video) -> List[Scene]:
        """
        Split video into coarse OCR-defined scenes.

        Returns:
            List[Scene]: One scene per detected OCR segment.
        """
        starts = self._splitter.split(video)
        fps = video.sequence.fps
        duration = video.audio.duration

        scenes: List[Scene] = []

        for i, s in enumerate(starts):
            start_t = s["frame_idx"] / fps
            end_t = starts[i + 1]["frame_idx"] / fps if i + 1 < len(starts) else duration

            scenes.append(
                Scene(
                    start_time=start_t,
                    end_time=end_t,
                    label=f"boulder_{s['boulder']}" if s.get("boulder") is not None else None,
                    metadata={"source": "ocr"},
                )
            )

        return scenes
