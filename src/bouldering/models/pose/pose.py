"""MediaPipe Pose model wrapper.

This module provides an inference-only wrapper around MediaPipe Pose.
It estimates human body landmarks for a single person cropped from
a video frame.

This module must NOT:
- compute domain-specific features
- aggregate temporally
- make decisions (events, scoring)

It ONLY converts images -> pose landmarks.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.bouldering.utils.typing import PoseLandmarks

# ---------------------------------------------------------------------
# MediaPipe Pose Wrapper
# ---------------------------------------------------------------------


class MediaPipePoseEstimator:
    """MediaPipe Pose inference wrapper.

    This class performs pose estimation on cropped person images.
    It returns pose landmarks in image coordinates, aligned with
    the original frame.

    Attributes:
        pose: MediaPipe Pose model instance.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize the MediaPipe Pose estimator.

        Args:
            static_image_mode: Whether to treat each image as independent.
                Set False for video processing.
            model_complexity: Pose model complexity (0, 1, or 2).
            smooth_landmarks: Whether to apply landmark smoothing.
            min_detection_confidence: Minimum detection confidence.
            min_tracking_confidence: Minimum tracking confidence.
        """
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils

        self.pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def estimate(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[PoseLandmarks]:
        """Estimate pose landmarks for a person in a frame.

        The frame is cropped using the provided bounding box before
        pose estimation. Returned landmarks are mapped back to the
        original frame coordinates.

        Args:
            frame: Full video frame in BGR format (OpenCV).
            bbox: Bounding box (x1, y1, x2, y2) of the person.

        Returns:
            PoseLandmarks or None if no pose is detected.
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Clamp bbox to image bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # MediaPipe expects RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        results = self.pose.process(crop_rgb)

        if not results.pose_landmarks:
            return None

        landmarks: PoseLandmarks = {}

        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            name = self._mp_pose.PoseLandmark(idx).name.lower()

            # Convert normalized coords -> image coords
            lx = x1 + landmark.x * (x2 - x1)
            ly = y1 + landmark.y * (y2 - y1)

            landmarks[name] = (lx, ly, landmark.visibility)

        return landmarks

    # -----------------------------------------------------------------
    # Optional utilities
    # -----------------------------------------------------------------

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.pose.close()
