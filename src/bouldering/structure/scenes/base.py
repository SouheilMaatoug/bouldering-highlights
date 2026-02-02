from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.bouldering.media.video.video import Video


@dataclass(frozen=True)
class Scene:
    """A class for defining a scene in a video (segment of a video).

    Attributes:
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        label (Optional[str]): Semantic label (e.g. 'boulder_3').
        metadata (Optional[dict]): Extra information.
    """

    start_time: float
    end_time: float
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SceneSplitter(ABC):
    """Abstract base class for scene splitters."""

    @abstractmethod
    def split(self, video: Video) -> List[Scene]:
        """Split a video into scenes.

        Args:
            video (Video): Input video.

        Returns:
            List[Scene]: List of detected scenes.
        """
        raise NotImplementedError
