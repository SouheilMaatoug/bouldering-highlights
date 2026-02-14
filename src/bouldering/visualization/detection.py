from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.bouldering.features.pose import center_of_gravity
from src.bouldering.utils.typing import BBox, PoseLandmarks, Tracks
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


def plot_frame_with_landmarks(
    frame,
    landmarks: PoseLandmarks,
    min_visibility: float = 0.5,
    figsize: tuple = (6, 5),
    title: Optional[str] = None,
):
    """Plot a single frame with pose landmarks and center of gravity.

    Landmarks are plotted in green.
    Center of gravity (CoG) is plotted in red.

    Args:
        frame (np.ndarray): BGR frame (OpenCV).
        landmarks (dict): landmark_name -> (x, y, visibility).
        min_visibility (float): minimum visibility threshold.
        figsize (tuple): matplotlib figure size.
        title (str | None): optional title.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb)
    ax.axis("off")

    if title:
        ax.set_title(title)

    # Plot landmarks (green)
    for x, y, v in landmarks.values():
        if v >= min_visibility:
            ax.scatter(x, y, c="lime", s=10)

    # Plot center of gravity (red)
    pose_frame = {"landmarks": landmarks}
    cog = center_of_gravity(pose_frame)
    if cog is not None:
        cx, cy = cog
        ax.scatter(cx, cy, c="red", s=80, marker="x", linewidths=2)

    plt.show()


def plot_frames_and_curve(
    video,
    pose_track: List[Dict],
    curve: List[Tuple[float, Optional[float]]],
    curve_name: str,
    n_samples: int = 5,
    min_visibility: float = 0.5,
    figsize: tuple = (18, 6),
):
    """Plot a sequence of sampled frames (top) with landmarks and CoG, and a single feature curve (bottom).

    Args:
        video: Video object.
        pose_track: List of pose frames.
        curve: List of (time, value) tuples.
        curve_name: Name of the curve.
        n_samples: Number of frames to sample.
        min_visibility: Landmark visibility threshold.
        figsize: Figure size.
    """
    # ------------------------------------------------------------
    # Prepare valid pose frames with CoG
    # ------------------------------------------------------------
    valid_frames = [f for f in pose_track if center_of_gravity(f) is not None]

    if len(valid_frames) < n_samples:
        raise ValueError("Not enough valid pose frames to sample.")

    # Sample frames uniformly
    sample_indices = np.linspace(0, len(valid_frames) - 1, n_samples).astype(int)

    sampled_frames = [valid_frames[i] for i in sample_indices]
    sample_times = [f["time"] for f in sampled_frames]

    # ------------------------------------------------------------
    # Create subplot layout
    # ------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, n_samples, height_ratios=[2.2, 1])

    frame_axes = [fig.add_subplot(gs[0, i]) for i in range(n_samples)]
    ax_curve = fig.add_subplot(gs[1, :])

    # ------------------------------------------------------------
    # Top row: frames + landmarks + CoG
    # ------------------------------------------------------------
    for ax, pose_frame in zip(frame_axes, sampled_frames):
        frame_idx = pose_frame["frame"]
        landmarks = pose_frame["landmarks"]
        t = pose_frame["time"]

        frame = video.sequence.frame(frame_idx)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(f"t = {t:.2f}s")

        # Plot landmarks (green)
        for x, y, v in landmarks.values():
            if v >= min_visibility:
                ax.scatter(x, y, c="lime", s=10)

        # Plot CoG (red)
        cog = center_of_gravity(pose_frame)
        if cog is not None:
            cx, cy = cog
            ax.scatter(cx, cy, c="red", s=60, marker="x", linewidths=2)

    # ------------------------------------------------------------
    # Bottom row: single curve
    # ------------------------------------------------------------
    xs = [t for t, v in curve if v is not None]
    ys = [v for t, v in curve if v is not None]

    ax_curve.plot(xs, ys, label=curve_name)

    # Mark sampled frames on curve
    sampled_values = []
    for t in sample_times:
        val = next((v for tt, v in curve if tt == t), None)
        if val is not None:
            sampled_values.append((t, val))

    if sampled_values:
        ax_curve.scatter(
            [t for t, _ in sampled_values],
            [v for _, v in sampled_values],
            color="red",
            zorder=5,
            label="Sampled frames",
        )

    ax_curve.set_xlabel("Time (s)")
    ax_curve.set_title(curve_name)
    ax_curve.grid(True)
    ax_curve.legend()

    # Invert Y-axis for image-space vertical signals
    # if "y" in curve_name.lower() or "gravity" in curve_name.lower():
    #   ax_curve.invert_yaxis()

    plt.tight_layout()
    plt.show()
