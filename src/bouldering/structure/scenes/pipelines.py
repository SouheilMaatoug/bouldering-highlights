from typing import List

from src.bouldering.media.video.video import Video
from src.bouldering.structure.scenes.base import Scene
from src.bouldering.structure.scenes.content import ContentSplitter
from src.bouldering.structure.scenes.ocr import OCRSplitter


class ScenePipeline:
    """Hierarchical scene detection pipeline.

    1. Macro splitting using OCR (semantic)
    2. Micro splitting using content changes (visual)
    """

    def __init__(
        self,
        macro_splitter: OCRSplitter,
        micro_splitter: ContentSplitter,
    ):
        """Initialize the hierarchical scene detection pipeline.

        Args:
            macro_splitter (OCRSplitter): OCR-based splitter.
            micro_splitter (ContentSplitter): Component-based splitter.
        """
        self.macro = macro_splitter
        self.micro = micro_splitter

    def run(self, video: Video) -> List[Scene]:
        """Run the full scene detection pipeline.

        Args:
            video (Video): Input video.

        Returns:
            List[Scene]: Final refined scenes.
        """
        final_scenes: List[Scene] = []

        macro_scenes = self.macro.split(video)

        for macro in macro_scenes:
            sub_scenes = self.micro.split(video, macro.start_time, macro.end_time)

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
