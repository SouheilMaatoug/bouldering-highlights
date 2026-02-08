"""Bouldering event detection.

This module detects semantic bouldering events from time-aware
visual feature signals.

Important:
- No pose estimation here
- No feature computation here
- No scoring here
- Events are temporal (with duration), not per-frame
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

TimeValue = Tuple[float, Optional[float]]
Event = Dict[str, object]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def contiguous_segments(
    times: List[float],
    max_gap: float,
) -> List[List[float]]:
    """Split timestamps into contiguous segments."""
    if not times:
        return []

    segments = [[times[0]]]
    for t in times[1:]:
        if t - segments[-1][-1] <= max_gap:
            segments[-1].append(t)
        else:
            segments.append([t])

    return segments


# ---------------------------------------------------------------------
# FALL detection
# ---------------------------------------------------------------------


def detect_fall(
    velocity_norm: List[Tuple[float, Optional[float]]],
    cog_y_norm: List[Tuple[float, Optional[float]]],
    visibility: List[Tuple[float, Optional[float]]],
    velocity_threshold: float = -0.3,
    ground_threshold: float = 0.15,
    min_duration: float = 0.25,
    max_gap: float = 0.15,
) -> List[Dict]:
    """Detect FALL events.

    Args:
        velocity_norm: (time, normalized vertical velocity)
        cog_y_norm: (time, normalized CoG Y)
        visibility: (time, pose visibility ratio)
        velocity_threshold: downward velocity threshold
        ground_threshold: CoG near ground threshold
        min_duration: minimum fall duration (seconds)
        max_gap: max gap between samples

    Returns:
        List of FALL event dictionaries.
    """
    candidate_times = []

    for (t, v), (_, y), (_, vis) in zip(
        velocity_norm,
        cog_y_norm,
        visibility,
    ):
        if v is not None and y is not None and v < velocity_threshold and y < ground_threshold:
            candidate_times.append(t)

    segments = contiguous_segments(candidate_times, max_gap)
    events = []

    for seg in segments:
        duration = seg[-1] - seg[0]
        if duration < min_duration:
            continue

        vis_after = [v for (t, v) in visibility if t >= seg[0] and v is not None]

        confidence = 1.0 - min(vis_after) if vis_after else 0.5

        events.append(
            {
                "type": "FALL",
                "start": seg[0],
                "end": seg[-1],
                "duration": duration,
                "confidence": confidence,
            }
        )

    return events


# ---------------------------------------------------------------------
# DYNO detection
# ---------------------------------------------------------------------


def detect_dyno(
    velocity_norm: List[Tuple[float, Optional[float]]],
    acceleration_norm: List[Tuple[float, Optional[float]]],
    displacement_amplitude_norm: List[Tuple[float, Optional[float]]],
    min_velocity: float = 0.25,
    min_acceleration: float = 2.0,
    min_amplitude: float = 0.08,
    max_duration: float = 0.6,
    max_gap: float = 0.12,
) -> List[Dict]:
    """Detect DYNO (dynamic jump) events.

    Args:
        velocity_norm: (time, normalized vertical velocity)
        acceleration_norm: (time, normalized vertical acceleration)
        displacement_amplitude_norm: (time, normalized displacement amplitude)
        min_velocity: minimum upward velocity
        min_acceleration: minimum upward acceleration
        min_amplitude: minimum displacement amplitude
        max_duration: maximum dyno duration
        max_gap: maximum gap between samples

    Returns:
        List of DYNO event dictionaries.
    """
    candidate_times = []

    for (t, v), (_, a), (_, amp) in zip(
        velocity_norm,
        acceleration_norm,
        displacement_amplitude_norm,
    ):
        if (
            v is not None
            and a is not None
            and amp is not None
            and v > min_velocity
            and a > min_acceleration
            and amp > min_amplitude
        ):
            candidate_times.append(t)

    segments = contiguous_segments(candidate_times, max_gap)
    events = []

    for seg in segments:
        duration = seg[-1] - seg[0]

        if duration > max_duration:
            continue

        events.append(
            {
                "type": "DYNO",
                "start": seg[0],
                "end": seg[-1],
                "duration": duration,
                "confidence": min(
                    1.0,
                    max(seg) - min(seg) if len(seg) > 1 else 0.7,
                ),
            }
        )

    return events


# ---------------------------------------------------------------------
# TOP detection
# ---------------------------------------------------------------------


def detect_top(
    hands_above_shoulders: List[Tuple[float, Optional[bool]]],
    cog_y_norm: List[Tuple[float, Optional[float]]],
    motion_energy: List[Tuple[float, Optional[float]]],
    top_threshold: float = 0.85,
    max_motion: float = 0.08,
    min_duration: float = 1.0,
    max_gap: float = 0.2,
) -> List[Dict]:
    """Detect TOP (finish) events.

    Args:
        hands_above_shoulders: (time, bool)
        cog_y_norm: (time, normalized CoG Y)
        motion_energy: (time, motion magnitude)
        top_threshold: CoG near top threshold
        max_motion: maximum allowed motion
        min_duration: minimum TOP duration
        max_gap: max gap between samples

    Returns:
        List of TOP event dictionaries.
    """
    candidate_times = []

    for (t, h), (_, y), (_, m) in zip(
        hands_above_shoulders,
        cog_y_norm,
        motion_energy,
    ):
        if h and y is not None and y > top_threshold and m is not None and m < max_motion:
            candidate_times.append(t)

    segments = contiguous_segments(candidate_times, max_gap)
    events = []

    for seg in segments:
        duration = seg[-1] - seg[0]
        if duration < min_duration:
            continue

        events.append(
            {
                "type": "TOP",
                "start": seg[0],
                "end": seg[-1],
                "duration": duration,
                "confidence": 1.0,
            }
        )

    return events


# ---------------------------------------------------------------------
# CRUX detection
# ---------------------------------------------------------------------


def detect_crux(
    motion_variance: List[Tuple[float, Optional[float]]],
    vertical_progress: List[Tuple[float, Optional[float]]],
    min_duration: float = 2.5,
    max_progress: float = 0.15,
    max_gap: float = 0.3,
) -> List[Dict]:
    """Detect CRUX (hard section) events.

    Args:
        motion_variance: (time, motion variance)
        vertical_progress: (time, vertical displacement)
        min_duration: minimum crux duration
        max_progress: maximum allowed vertical progress
        max_gap: max gap between samples

    Returns:
        List of CRUX event dictionaries.
    """
    # Determine dynamic threshold (75th percentile)
    values = [v for _, v in motion_variance if v is not None]
    if not values:
        return []

    variance_threshold = np.percentile(values, 75)

    candidate_times = [t for (t, v) in motion_variance if v is not None and v > variance_threshold]

    segments = contiguous_segments(candidate_times, max_gap)
    events = []

    for seg in segments:
        duration = seg[-1] - seg[0]
        if duration < min_duration:
            continue

        progress = [p for (t, p) in vertical_progress if seg[0] <= t <= seg[-1] and p is not None]

        if progress and max(progress) < max_progress:
            events.append(
                {
                    "type": "CRUX",
                    "start": seg[0],
                    "end": seg[-1],
                    "duration": duration,
                    "confidence": float(np.mean(progress)),
                }
            )

    return events
