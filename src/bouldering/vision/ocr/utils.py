import cv2
import numpy as np


def crop_frame_percent(frame: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Crop frame using relative coordinates."""
    h, w = frame.shape[:2]
    return frame[
        int(y0 * h) : int(y1 * h),
        int(x0 * w) : int(x1 * w),
    ]


def resize(frame: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Resize frame by scale factors."""
    return cv2.resize(frame, None, fx=fx, fy=fy)
