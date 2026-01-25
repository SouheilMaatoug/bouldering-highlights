import cv2
import numpy as np


def crop_frame_percent(
    frame: np.ndarray, x0_p: float, y0_p: float, x1_p: float, y1_p: float
) -> np.ndarray:
    """Crop an image using a bounding box.

    Args:
        frame (np.ndarray): The image frame.
        x0_p (float): The x coordinate of the first corner in percentage of the width. In [0, 1]
        y0_p (float): The y coordinate of the first corner in percentage of the height. In [0, 1]
        x1_p (float): The x coordinate of the second corner in percentage of the width. In [0, 1]
        y1_p (float): The y coordinate of the second corner in percentage of the height. In [0, 1]

    Returns:
        np.ndarray: The crop image.
    """
    h, w = frame.shape[:2]
    x0 = int(x0_p * w)
    x1 = int(x1_p * w)
    y0 = int(y0_p * h)
    y1 = int(y1_p * h)
    return frame[y0:y1, x0:x1]


def resize(frame: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Resize in image by setting x and y factors.

    Args:
        frame (np.ndarray): The image frame.
        fx (float): The width factor.
        fy (float): The height factor.

    Returns:
        np.ndarray: The resized image.
    """
    return cv2.resize(frame, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
