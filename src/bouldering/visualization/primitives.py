from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.bouldering.utils.typing import BBox


def prepare_figure(
    frame: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
):
    """Prepare a Matplotlib figure for an OpenCV frame.

    This function converts a BGR OpenCV frame to RGB, initializes a Matplotlib
    figure and axis, and optionally sets a title.

    Args:
        frame: Input image in BGR format (OpenCV).
        figsize: Size of the Matplotlib figure. Defaults to (10, 6).
        title: Optional title to display above the frame.

    Returns:
        tuple:
            - fig: Matplotlib Figure object
            - ax: Matplotlib Axes object
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb)
    ax.axis("off")

    if title:
        ax.set_title(title)

    return fig, ax


def draw_bbox(
    ax,
    bbox: BBox,
    label: Optional[str] = None,
    color: str = "lime",
    linewidth: int = 2,
) -> None:
    """Draw a bounding box with an optional label on an axis.

    Args:
        ax: Matplotlib Axes on which to draw.
        bbox: Bounding box to draw.
        label: Optional text label (e.g., track ID).
        color: Color of the bounding box and label.
        linewidth: Thickness of the bounding box lines.
    """
    rect = plt.Rectangle(
        (bbox.x1, bbox.y1),
        bbox.width,
        bbox.height,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
    )
    ax.add_patch(rect)

    if label:
        ax.text(
            bbox.x1,
            bbox.y1 - 5,
            label,
            color=color,
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, pad=2),
        )
