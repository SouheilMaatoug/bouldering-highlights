from typing import List, Tuple

import numpy as np

from src.bouldering.utils.typing import TrackEntry, Tracks


def temporal_gap(track_a: List[TrackEntry], track_b: List[TrackEntry]) -> float:
    """Detect the temporal gap between two tracks.

    Args:
        track_a (List[TrackEntry]): Track a.
        track_b (List[TrackEntry]): Track b.

    Returns:
        float: Time gap in seconds.
    """
    end_a = track_a[-1]["frame"]
    start_b = track_b[0]["frame"]
    return start_b - end_a


def spatial_distance(track_a: List[TrackEntry], track_b: List[TrackEntry], frame_shape: Tuple[int, int, int]) -> float:
    """Detect the spatial gap between two tracks (relative distance).

    Args:
        track_a (List[TrackEntry]): Track a.
        track_b (List[TrackEntry]): Track b
        frame_shape (Tuple[int, int, int]): Frame shape.

    Returns:
        float: Spatial distance in pixels.
    """
    c1 = track_a[-1]["bbox"].center
    c2 = track_b[0]["bbox"].center
    h, w = frame_shape[:2]
    return np.linalg.norm((c1 - c2) / np.array([w, h]))


# merge decision rule
def should_merge(
    track_a: List[TrackEntry],
    track_b: List[TrackEntry],
    frame_shape: Tuple[int, int, int],
    max_gap: float = 100,
    max_distance: float = 0.5,
) -> bool:
    """Track merging decision.

    Args:
        track_a (List[TrackEntry]): Track a.
        track_b (List[TrackEntry]): Track b.
        frame_shape (Tuple[int, int, int]): Frame shape.
        max_gap (float, optional): Maximum temporal gap. Defaults to 100.
        max_distance (float, optional): Maximum spatial distance. Defaults to 0.5.

    Returns:
        bool: merging decision.
    """
    gap = temporal_gap(track_a, track_b)
    if not (0 < gap <= max_gap):
        return False

    dist = spatial_distance(track_a, track_b, frame_shape)
    if dist > max_distance:
        return False

    return True


# postprocessing merge tracks pipeline
def merge_tracks(tracks: Tracks, frame_shape: Tuple[int, int, int]) -> Tracks:
    """Merge tracks if they correspond to the same person.

    Args:
        tracks (Tracks): Detected tracks to be potentially merged.
        frame_shape (Tuple[int, int, int]): Frame shape.

    Returns:
        Tracks: The merged tracks.
    """
    track_items = list(tracks.items())
    merged = {}
    used = set()
    new_id = 0

    for i, (id_a, track_a) in enumerate(track_items):
        if id_a in used:
            continue

        current = track_a.copy()
        used.add(id_a)

        for id_b, track_b in track_items:
            if id_b in used:
                continue

            if should_merge(current, track_b, frame_shape):
                current.extend(track_b)
                used.add(id_b)

        merged[new_id] = sorted(current, key=lambda x: x["frame"])
        new_id += 1

    return merged
