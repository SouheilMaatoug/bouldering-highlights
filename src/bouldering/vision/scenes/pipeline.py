from typing import List

from src.bouldering.media.video.video import Video
from src.bouldering.vision.scenes.base import Scene
from src.bouldering.vision.scenes.content import ContentSceneSplitter
from src.bouldering.vision.scenes.ocr import OCRSceneSplitter


class ScenePipeline:
    """
    Hierarchical scene detection pipeline.

    1. Macro splitting using OCR (semantic)
    2. Micro splitting using content changes (visual)
    """

    def __init__(
        self,
        macro_splitter: OCRSceneSplitter,
        micro_splitter: ContentSceneSplitter,
    ):
        self.macro = macro_splitter
        self.micro = micro_splitter

    def run(self, video: Video) -> List[Scene]:
        """
        Run the full scene detection pipeline.

        Returns:
            List[Scene]: Final refined scenes.
        """
        final_scenes: List[Scene] = []

        macro_scenes = self.macro.split(video)

        for macro in macro_scenes:
            sub_scenes = self.micro.split_range(video, macro.start_time, macro.end_time)

            if not sub_scenes:
                final_scenes.append(macro)
                continue

            for sc in sub_scenes:
                final_scenes.append(
                    Scene(
                        start_time=sc.start_time,
                        end_time=sc.end_time,
                        label=macro.label,
                        metadata={
                            "parent": macro,
                            "source": "ocr+content",
                        },
                    )
                )

        return final_scenes
