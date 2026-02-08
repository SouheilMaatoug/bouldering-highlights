from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, TypedDict

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box.

    Coordinates follow the (x1, y1, x2, y2) convention in image space.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Center (cx, cy) of the bounding box."""
        return np.array(
            [
                (self.x1 + self.x2) / 2.0,
                (self.y1 + self.y2) / 2.0,
            ]
        )

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        """Return the bounding box as an (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def iou(self, other: "BBox") -> float:
        """Compute Intersection-over-Union with another bounding box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih

        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


class TrackEntry(TypedDict):
    """Single observation of a track at one frame."""

    frame: int
    bbox: BBox
    confidence: float


Tracks = Dict[int, List[TrackEntry]]


class Event(TypedDict, total=False):
    """Semantic bouldering event."""

    type: str  # e.g. "FALL", "DYNO", "CRUX", "TOP"
    frame: int
    duration: float
    intensity: float
    score: float


Landmark = Tuple[float, float, float]  # (x, y, visibility)
PoseLandmarks = Dict[str, Landmark]
PoseFrame = Dict[str, object]
