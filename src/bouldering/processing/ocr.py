from typing import Any, Dict, List, Optional

import easyocr
import numpy as np
import torch
from tqdm import tqdm

from src.bouldering.media.video.video import Video
from src.bouldering.processing import utils


class OCRDetector:
    """Class for detecting with OCR."""

    def __init__(
        self, langs: list[str] = ["en"], use_gpu: Optional[bool] = None, decoder: str = "greedy"
    ):
        """Initialize the OCRDetector class.

        Args:
            langs (list[str], optional): The list of langages to detect. Defaults to english ["en"].
            use_gpu (bool, optional): Use GPU for detection. Defaults to None.
            decoder (str, optional): The decoder to use. options are 'greedy', 'beamsearch' and 'wordbeamsearch'.
                Defaults to "greedy".
        """
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(langs, gpu=use_gpu)
        self.decoder = decoder

    def read_batch(self, batch_images: List[np.ndarray]) -> List[Dict]:
        """Read on a batch of images.

        Args:
            batch_images (List[np.ndarray]): The batch of images.

        Returns:
            List[Dict]: The list of results.
        """
        return self.reader.readtext_batched(batch_images, detail=1, decoder=self.decoder)

    def run(
        self,
        video: Video,
        crop_box: List[int],
        fx: float,
        fy: float,
        min_conf: float = 0.0,
        stride: int = 3,
        batch_size: int = 16,
    ) -> List[Dict[int, Any]]:
        """Run the OCR detector on a video input with preprocessing.

        Args:
            video (Video): The input video.
            crop_box (List[int]): The list of relative coordinates of the bounding box. [x0, y0, x1, y1].
            fx (float): Width resize factor.
            fy (float): Height resize factor.
            min_conf (float, optional): Minimum confidence in detections. Defaults to 0.0.
            stride (int, optional): The stride to advance in detections. Defaults to 3.
            batch_size (int, optional): The inference batch size. Defaults to 16.

        Returns:
            List[Dict[int, Any]]: A list of dictionaries having frame indices as keys and ocr detections as values.
        """
        x0, y0, x1, y1 = crop_box

        frames = video.sequence.frames()
        n_frames = video.metadata["video"]["n_frames"]
        pbar = tqdm(total=n_frames, desc="Running batched OCR")

        frame_outputs = {}

        batch_imgs = []
        batch_indices = []
        idx = -1

        def flush():
            if not batch_imgs:
                return
            results = self.read_batch(batch_imgs)
            for fidx, items in zip(batch_indices, results):
                frame_outputs.setdefault(fidx, [])
                if not items:
                    continue
                for bbox, text, conf in items:
                    if conf is None or conf < min_conf:
                        continue
                    frame_outputs[fidx].append({"text": (text or "").strip(), "conf": float(conf)})
            batch_imgs.clear()
            batch_indices.clear()

        while True:
            try:
                idx += 1
                frame = next(frames)
            except StopIteration:
                flush()
                break

            if idx % stride != 0:
                continue

            # First sampled frame defines a default ROI helper if not given
            cropped = utils.crop_frame_percent(frame, x0, y0, x1, y1)
            resized = utils.resize(cropped, fx, fy)
            batch_imgs.append(resized)
            batch_indices.append(idx)

            if len(batch_imgs) == batch_size:
                flush()
            pbar.update(1)

        return frame_outputs
