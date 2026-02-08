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
    cog_velocity: List[TimeValue],
    visibility: List[TimeValue],
    velocity_threshold: float = -0.8,
    min_duration: float = 0.2,
    max_gap: float = 0.15,
) -> List[Event]:
    """Detect fall events.

    Fall definition:
    - strong downward CoG velocity
    - followed or accompanied by visibility drop

    Args:
        cog_velocity: (time, vertical velocity)
        visibility: (time, pose visibility ratio)
        velocity_threshold: downward velocity threshold
        min_duration: minimum fall duration (seconds)
        max_gap: max allowed gap between samples

    Returns:
        List of FALL events.
    """
    fall_times = [t for (t, v) in cog_velocity if v is not None and v < velocity_threshold]

    segments = contiguous_segments(fall_times, max_gap)
    events: List[Event] = []

    for seg in segments:
        duration = seg[-1] - seg[0]
        if duration < min_duration:
            continue

        # Check visibility drop after fall start
        vis_after = [v for (t, v) in visibility if t >= seg[0] and v is not None]

        if vis_after and min(vis_after) < 0.4:
            events.append(
                {
                    "type": "FALL",
                    "start": seg[0],
                    "end": seg[-1],
                    "duration": duration,
                    "confidence": float(min(vis_after)),
                }
            )

    return events


# ---------------------------------------------------------------------
# DYNO detection
# ---------------------------------------------------------------------


def detect_dyno(
    cog_velocity: List[TimeValue],
    cog_acceleration: List[TimeValue],
    min_acceleration: float = 2.5,
    min_velocity: float = 1.0,
    max_duration: float = 0.6,
) -> List[Event]:
    """Detect dyno (dynamic jump) events.

    Dyno definition:
    - strong upward acceleration
    - followed by upward velocity spike
    - short duration

    Args:
        cog_velocity: (time, vertical velocity)
        cog_acceleration: (time, vertical acceleration)
        min_acceleration: minimum acceleration threshold.
        min_velocity: minimum velocity threshold.
        max_duration: maximum duration.

    Returns:
        List of DYNO events.
    """
    dyno_times = [t for (t, a) in cog_acceleration if a is not None and a > min_acceleration]

    events: List[Event] = []

    for t in dyno_times:
        vel_after = [v for (t2, v) in cog_velocity if t <= t2 <= t + max_duration and v is not None]

        if vel_after and max(vel_after) > min_velocity:
            events.append(
                {
                    "type": "DYNO",
                    "start": t,
                    "end": t + max_duration,
                    "duration": max_duration,
                    "confidence": max(vel_after),
                }
            )

    return events


# ---------------------------------------------------------------------
# TOP detection
# ---------------------------------------------------------------------


def detect_top(
    hands_above_head: List[TimeValue],
    motion_energy: List[TimeValue],
    min_duration: float = 1.0,
    max_motion: float = 0.15,
) -> List[Event]:
    """Detect TOP (finish) events.

    Top definition:
    - hands above head
    - low motion
    - sustained

    Args:
        hands_above_head: (time, bool)
        motion_energy: (time, motion magnitude)
        min_duration: minimum duration threshold.
        max_motion: maximum motion.

    Returns:
        List of TOP events.
    """
    valid_times = [t for (t, h) in hands_above_head if h]

    segments = contiguous_segments(valid_times, max_gap=0.2)
    events: List[Event] = []

    for seg in segments:
        duration = seg[-1] - seg[0]
        if duration < min_duration:
            continue

        motion = [m for (t, m) in motion_energy if seg[0] <= t <= seg[-1] and m is not None]

        if motion and max(motion) < max_motion:
            events.append(
                {
                    "type": "TOP",
                    "start": seg[0],
                    "end": seg[-1],
                    "duration": duration,
                    "confidence": 1.0 - max(motion),
                }
            )

    return events


# ---------------------------------------------------------------------
# CRUX detection
# ---------------------------------------------------------------------


def detect_crux(
    motion_variance: List[TimeValue],
    vertical_progress: List[TimeValue],
    min_duration: float = 3.0,
    max_progress: float = 0.2,
) -> List[Event]:
    """Detect crux (hard section) events.

    Crux definition:
    - high motion variance
    - low vertical progress
    - long duration

    Args:
        motion_variance: (time, variance)
        vertical_progress: (time, vertical displacement)
        min_duration: minimum event duration.
        max_progress: maximum progress.

    Returns:
        List of CRUX events.
    """
    candidate_times = [
        t
        for (t, mv) in motion_variance
        if mv is not None and mv > np.percentile([v for _, v in motion_variance if v is not None], 75)
    ]

    segments = contiguous_segments(candidate_times, max_gap=0.3)
    events: List[Event] = []

    for seg in segments:
        duration = seg[-1] - seg[0]
        if duration < min_duration:
            continue

        progress = [p for (t, p) in vertical_progress if seg[0] <= t <= seg[-1] and p is not None]

        if progress and (max(progress) - min(progress)) < max_progress:
            events.append(
                {
                    "type": "CRUX",
                    "start": seg[0],
                    "end": seg[-1],
                    "duration": duration,
                    "confidence": np.mean(progress),
                }
            )

    return events
