from collections import defaultdict
from typing import Iterable

import numpy as np
import supervision as sv
from tqdm import tqdm

from src.bouldering.models.detection.yolo import YoloPersonDetector
from src.bouldering.utils.typing import BBox, TrackEntry, Tracks


class YoloPersonTracker:
    """ByteTrack-based multi-person tracker based on Yolo detections.

    This class performs tracking inference by combining:
    - a person detector (YOLO)
    - ByteTrack for temporal identity association

    It is inference-only:
    - no ID stitching
    - no active competitor detection
    - no domain logic
    """

    def __init__(
        self,
        detector: YoloPersonDetector,
        fps: int,
        lost_track_buffer: int = 50,
        track_activation_threshold: float = 0.2,
        minimum_matching_threshold: float = 0.8,
    ) -> None:
        """Initialize the YoloPersonTracker.

        Args:
            detector: Person detector instance.
            fps: Frame rate of the video.
            lost_track_buffer: Number of frames a track is kept alive
                without detections. Defaults to 50.
            track_activation_threshold: Minimum confidence required
                to activate a new track. Defaults to 0.2.
            minimum_matching_threshold: Matching threshold between
                detections and existing tracks. Defaults to 0.8.
        """
        self.detector: YoloPersonDetector = detector

        self.tracker: sv.ByteTrack = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=fps,
            minimum_consecutive_frames=1,
        )

    def track_frames(self, frames: Iterable[np.ndarray]) -> Tracks:
        """Track persons across a sequence of frames.

        Args:
            frames: Iterable of BGR frames (NumPy arrays).

        Returns:
            Tracks: Mapping from track ID to a list of track entries.
            Each track entry contains:
                - "frame": frame index
                - "bbox": (x1, y1, x2, y2)
                - "confidence": detection confidence
        """
        tracks: Tracks = defaultdict(list)

        for frame_idx, frame in tqdm(enumerate(frames)):
            detections: sv.Detections = self.detector.detect(frame)
            tracked: sv.Detections = self.tracker.update_with_detections(detections)

            for xyxy, track_id, conf in zip(
                tracked.xyxy,
                tracked.tracker_id,
                tracked.confidence,
            ):
                x1, y1, x2, y2 = map(float, xyxy)

                tracks[int(track_id)].append(
                    TrackEntry(
                        frame=frame_idx,
                        bbox=BBox(x1, y1, x2, y2),
                        confidence=float(conf),
                    )
                )

        return tracks
