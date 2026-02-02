from typing import List

from src.bouldering.media.video.video import Video
from src.bouldering.models.ocr.splitter import SceneSplitterOCR
from src.bouldering.structure.scenes.base import Scene, SceneSplitter


class OCRSplitter(SceneSplitter):
    """Scene splitter based on OCR detection ('BOULDERING')."""

    def __init__(self, splitter: SceneSplitterOCR):
        """Initialize the OCR-based scene splitter.

        Args:
            splitter (SceneSplitterOCR): OCR splitter object.
        """
        self._splitter = splitter

    def split(self, video: Video) -> List[Scene]:
        """Split a video into coarse OCR-defined scenes.

        Args:
            video (Video): Input video.

        Returns:
            List[Scene]: A list of scene objects.
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
