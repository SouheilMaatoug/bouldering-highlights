from collections import defaultdict
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.bouldering.models.detection.typing import BBox, Tracks
from src.bouldering.visualization.primitives import draw_bbox, prepare_figure


def plot_frame(
    frame: np.ndarray,
    bboxes: Optional[Iterable[BBox]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """Plot a single video frame with optional bounding boxes.

    Args:
        frame: Input frame in BGR format (OpenCV).
        bboxes: Optional iterable of bounding boxes to draw.
        title: Optional title displayed above the frame.
        figsize: Size of the Matplotlib figure.
    """
    _, ax = prepare_figure(frame, figsize=figsize, title=title)

    if bboxes:
        for bbox in bboxes:
            draw_bbox(ax, bbox)

    plt.show()


def plot_detections(
    frame: np.ndarray,
    detections: Iterable[BBox],
    title: Optional[str] = None,
) -> None:
    """Plot raw detection bounding boxes on a frame.

    Args:
        frame: Input frame in BGR format (OpenCV).
        detections: Iterable of detected bounding boxes.
        title: Optional title displayed above the frame.
    """
    _, ax = prepare_figure(frame, title=title)

    for bbox in detections:
        draw_bbox(ax, bbox)

    plt.show()


def plot_tracking_samples(
    video,
    tracks: Tracks,
    n_samples: int = 10,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """Plot tracking results on uniformly sampled frames from a video.

    This function is intended for debugging and manual validation of
    tracking quality (ID stability, missed detections, etc.).

    Args:
        video: Video object providing access to frames and frame count.
        tracks: Tracking results (track_id -> list of TrackEntry).
        n_samples: Number of frames to sample uniformly across the video.
        figsize: Size of the Matplotlib figure.
    """
    n_frames = video.sequence.n_frames
    frame_indices = np.linspace(0, n_frames - 1, n_samples).astype(int)

    # Build lookup: frame_idx -> [(track_id, bbox)]
    per_frame = defaultdict(list)
    for track_id, entries in tracks.items():
        for e in entries:
            per_frame[e["frame"]].append((track_id, e["bbox"]))

    for frame_idx in frame_indices:
        frame = video.sequence.frame(frame_idx)
        _, ax = prepare_figure(
            frame,
            figsize=figsize,
            title=f"Frame {frame_idx}",
        )

        for track_id, bbox in per_frame.get(frame_idx, []):
            draw_bbox(ax, bbox, label=f"ID {track_id}")

        plt.show()


def plot_tracks_on_frame(
    frame,
    tracks_at_frame: Iterable[Tuple[int, BBox]],
    title: str | None = None,
) -> None:
    """Plot tracked persons on a single frame.

    Args:
        frame: Input frame in BGR format (OpenCV).
        tracks_at_frame: Iterable of (track_id, bounding box) tuples.
        title: Optional title displayed above the frame.
    """
    _, ax = prepare_figure(frame, title=title)

    for track_id, bbox in tracks_at_frame:
        draw_bbox(ax, bbox, label=f"ID {track_id}")

    plt.show()
