import cv2
import numpy as np


def crop_frame_percent(frame: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Crop frame using relative coordinates.

    Args:
        frame (np.ndarray): Input image in HxWxC format.
        x0 (float): Left relative coordinate.
        y0 (float): Top relative coordinate.
        x1 (float): Right relative coordinate.
        y1 (float): Botoom relative coordinate.

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = frame.shape[:2]
    return frame[
        int(y0 * h) : int(y1 * h),
        int(x0 * w) : int(x1 * w),
    ]


def resize(frame: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Resize frame by scale factors.

    Args:
        frame (np.ndarray): Input image in HxWxC format.
        fx (float): Horizontal scale factor.
        fy (float): Vertical scale factor.

    Returns:
        np.ndarray: Resize image.
    """
    return cv2.resize(frame, None, fx=fx, fy=fy)
