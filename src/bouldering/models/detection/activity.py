"""Active track detection utilities.

This module contains generic, track-level temporal reasoning used to
identify which tracked persons are "active" in a scene. Activity is
defined purely from motion and presence over time, without any
domain-specific assumptions (e.g. bouldering).

This module must remain independent of:
- pose estimation
- sport-specific rules
- scoring logic
"""

from typing import List, Tuple

import numpy as np

from bouldering.models.typing import TrackEntry, Tracks

# ---------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------


def track_centers(track: List[TrackEntry]) -> List[Tuple[int, float, float]]:
    """Extract center positions from a track.

    Args:
        track: List of track entries.

    Returns:
        List of tuples (frame_index, cx, cy).
    """
    centers: List[Tuple[int, float, float]] = []

    for entry in track:
        cx, cy = entry["bbox"].center
        centers.append((entry["frame"], cx, cy))

    return centers


# ---------------------------------------------------------------------
# Motion & presence metrics
# ---------------------------------------------------------------------


def motion_score(
    track: List[TrackEntry],
    frame_shape: Tuple[int, int, int],
    fps: int,
    window_seconds: float = 5.0,
) -> float:
    """Compute a normalized motion score for a track.

    Motion is defined as the mean per-frame displacement of the track's
    bounding box center over a sliding temporal window.

    Args:
        track: List of track entries.
        frame_shape: Shape of the video frames (H, W, C).
        fps: Frame rate of the video.
        window_seconds: Temporal window size (seconds).

    Returns:
        Normalized motion score (float).
    """
    centers = track_centers(track)
    if len(centers) < 2:
        return 0.0

    window_frames = int(fps * window_seconds)
    last_frame = centers[-1][0]

    # Keep only recent points
    window = [(f, x, y) for (f, x, y) in centers if f >= last_frame - window_frames]

    if len(window) < 2:
        return 0.0

    height, width = frame_shape[:2]
    motions: List[float] = []

    for i in range(1, len(window)):
        _, x1, y1 = window[i - 1]
        _, x2, y2 = window[i]

        dx = (x2 - x1) / width
        dy = (y2 - y1) / height
        motions.append(np.sqrt(dx * dx + dy * dy))

    return float(np.mean(motions)) if motions else 0.0


def presence_ratio(
    track: List[TrackEntry],
    fps: int,
    window_seconds: float = 5.0,
) -> float:
    """Compute the presence ratio of a track over a time window.

    Presence ratio measures how often a track is observed relative to
    the expected number of frames in the window.

    Args:
        track: List of track entries.
        fps: Frame rate of the video.
        window_seconds: Temporal window size (seconds).

    Returns:
        Presence ratio in [0, 1].
    """
    window_frames = int(fps * window_seconds)
    last_frame = track[-1]["frame"]

    frames_present = {entry["frame"] for entry in track if entry["frame"] >= last_frame - window_frames}

    if window_frames <= 0:
        return 0.0

    return len(frames_present) / window_frames


# ---------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------


def detect_active_competitors(
    tracks: Tracks,
    frame_shape: Tuple[int, int, int],
    fps: int,
    window_seconds: float = 5.0,
    min_motion: float = 0.008,
    min_presence: float = 0.3,
) -> Tracks:
    """Filter tracks to keep only active ones.

    A track is considered active if:
    - it exhibits sufficient motion over a temporal window
    - it is present in a sufficient fraction of frames in that window

    This function preserves the input track format to allow seamless
    chaining with visualization and downstream processing.

    Args:
        tracks: Mapping from track ID to list of track entries.
        frame_shape: Shape of the video frames (H, W, C).
        fps: Frame rate of the video.
        window_seconds: Temporal window size (seconds).
        min_motion: Minimum motion score threshold.
        min_presence: Minimum presence ratio threshold.

    Returns:
        Tracks containing only active track IDs.
    """
    active_tracks: Tracks = {}

    for track_id, track in tracks.items():
        if len(track) < 2:
            continue

        motion = motion_score(
            track=track,
            frame_shape=frame_shape,
            fps=fps,
            window_seconds=window_seconds,
        )

        presence = presence_ratio(
            track=track,
            fps=fps,
            window_seconds=window_seconds,
        )

        if motion >= min_motion and presence >= min_presence:
            active_tracks[track_id] = track

    return active_tracks
