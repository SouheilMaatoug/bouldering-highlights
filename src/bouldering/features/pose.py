"""Pose-based visual feature extraction.

This module converts MediaPipe pose landmarks into interpretable,
per-frame visual features that are robust to missing frames.

Important:
- No temporal aggregation here
- No event detection here
- No scoring here
"""

from typing import Optional, Tuple

import numpy as np

from src.bouldering.utils.typing import PoseFrame, PoseLandmarks

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def get_landmark_xy(
    landmarks: PoseLandmarks,
    name: str,
    min_visibility: float = 0.5,
) -> Optional[Tuple[float, float]]:
    """Safely extract (x, y) for a landmark if visible."""
    if name not in landmarks:
        return None

    x, y, v = landmarks[name]
    if v < min_visibility:
        return None

    return x, y


def mean_xy(points: list[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Mean of a list of (x, y) points."""
    if not points:
        return None

    xs, ys = zip(*points)
    return float(np.mean(xs)), float(np.mean(ys))


# ---------------------------------------------------------------------
# Core pose features
# ---------------------------------------------------------------------


def center_of_gravity(pose_frame: PoseFrame) -> Optional[Tuple[float, float]]:
    """Approximate body center of gravity (CoG).

    Uses hips and shoulders as a robust torso proxy.

    Returns:
        (x, y) in image coordinates, or None
    """
    landmarks = pose_frame["landmarks"]

    points = []
    for x, y, v in landmarks.values():
        if v >= 0.5:
            points.append((x, y))

    if not points:
        return None

    xs, ys = zip(*points)
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def cog_y(pose_frame: PoseFrame) -> Optional[float]:
    """Vertical position of the center of gravity."""
    cog = center_of_gravity(pose_frame)
    return cog[1] if cog else None


def wrist_mean_y(pose_frame: PoseFrame) -> Optional[float]:
    """Average vertical position of both wrists."""
    landmarks = pose_frame["landmarks"]

    points = []
    for name in ["left_wrist", "right_wrist"]:
        p = get_landmark_xy(landmarks, name)
        if p:
            points.append(p)

    if not points:
        return None

    return float(np.mean([p[1] for p in points]))


def hands_above_shoulders(pose_frame: PoseFrame) -> Optional[bool]:
    """Check if hands are above shoulders."""
    landmarks = pose_frame["landmarks"]

    wrists = []
    shoulders = []

    for name in ["left_wrist", "right_wrist"]:
        p = get_landmark_xy(landmarks, name)
        if p:
            wrists.append(p[1])

    for name in ["left_shoulder", "right_shoulder"]:
        p = get_landmark_xy(landmarks, name)
        if p:
            shoulders.append(p[1])

    if not wrists or not shoulders:
        return None

    return np.mean(wrists) < np.mean(shoulders)


def hands_above_head(pose_frame: PoseFrame) -> Optional[bool]:
    """Check if hands are above head."""
    landmarks = pose_frame["landmarks"]

    wrists = []
    heads = []

    for name in ["left_wrist", "right_wrist"]:
        p = get_landmark_xy(landmarks, name)
        if p:
            wrists.append(p[1])

    head = get_landmark_xy(landmarks, "nose")
    if head:
        heads.append(head[1])

    if not wrists or not heads:
        return None

    return np.mean(wrists) < np.mean(heads)


def pose_visibility_ratio(
    pose_frame: PoseFrame,
    min_visibility: float = 0.5,
) -> float:
    """Compute ratio of visible landmarks."""
    landmarks = pose_frame["landmarks"]

    if not landmarks:
        return 0.0

    visible = sum(1 for _, _, v in landmarks.values() if v >= min_visibility)
    return visible / len(landmarks)


def normalize_cog_y(
    y_pixel: float,
    frame_height: int,
) -> float:
    """Normalize vertical coordinate so that.

    Normalized as follows:
    - 0.0 = bottom of frame.
    - 1.0 = top of frame.
    """
    return 1.0 - (y_pixel / frame_height)
