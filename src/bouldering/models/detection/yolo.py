import numpy as np
import supervision as sv
from ultralytics import YOLO


class YoloPersonDetector:
    """YOLO-based person detector."""

    PERSON_CLASS_ID: int = 0  # COCO class id for "person"

    def __init__(
        self,
        model_path: str = "yolov8l.pt",
        conf: float = 0.25,
        imgsz: int = 1280,
    ) -> None:
        """Initialize the YOLO person detector.

        Args:
            model_path: Path to the YOLO model weights. Defaults to 'yolov8.pt'.
            conf: Confidence threshold for detections. Default to 0.25.
            imgsz: Image size used for inference. Defaults to 1280.
        """
        self.model: YOLO = YOLO(model_path)
        self.model.conf = conf
        self.imgsz: int = imgsz

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Detect persons in a single video frame.

        Args:
            frame: Input frame as a BGR NumPy array (OpenCV format).

        Returns:
            sv.Detections: Detections containing only persons.
        """
        result = self.model(frame, imgsz=self.imgsz, verbose=False)[0]
        detections: sv.Detections = sv.Detections.from_ultralytics(result)

        # Filter only person detections
        detections = detections[detections.class_id == self.PERSON_CLASS_ID]

        return detections
