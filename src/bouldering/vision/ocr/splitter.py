from collections import Counter
from typing import Any, Dict

from tqdm import tqdm

from src.bouldering.media.video.video import Video
from src.bouldering.vision.ocr import utils
from src.bouldering.vision.ocr.detector import OCRDetector
from src.bouldering.vision.ocr.parser import parse_boulder_number


class SceneSplitterOCR:
    """
    Scene splitter driven by OCR 'BOULDER N' overlays.
    """

    def __init__(
        self,
        crop_box: list,
        langs=("en",),
        fx=0.5,
        fy=0.5,
        stride=3,
        batch_size=16,
        smooth_window=5,
        majority_ratio=0.6,
        require_number=True,
    ):
        self.crop_box = crop_box
        self.fx = fx
        self.fy = fy
        self.stride = stride
        self.batch_size = batch_size
        self.smooth_window = smooth_window
        self.majority_ratio = majority_ratio
        self.require_number = require_number

        self.detector = OCRDetector(langs=list(langs))

    def split(self, video: Video):
        """
        Split video into OCR-confirmed Boulder sections.

        Returns:
            List[Dict]: [{"frame_idx": int, "boulder": Optional[int]}]
        """
        frames = video.sequence.frames()
        n_frames = video.sequence.n_frames

        detections = {}

        batch_imgs = []
        batch_idxs = []
        idx = -1

        pbar = tqdm(
            total=n_frames,
            desc="OCR scene splitting",
            unit="frame",
        )

        def flush():
            if not batch_imgs:
                return
            results = self.detector.read_batch(batch_imgs)
            for fidx, items in zip(batch_idxs, results):
                detections[fidx] = items
            batch_imgs.clear()
            batch_idxs.clear()

        for frame in frames:
            idx += 1
            pbar.update(1)

            if idx % self.stride != 0:
                continue

            cropped = utils.crop_frame_percent(frame, *self.crop_box)
            resized = utils.resize(cropped, self.fx, self.fy)

            batch_imgs.append(resized)
            batch_idxs.append(idx)

            if len(batch_imgs) >= self.batch_size:
                flush()

        # flush remaining
        flush()
        pbar.close()

        return self._postprocess(detections)

    def _postprocess(self, detections: Dict[int, Any]):
        """Temporal smoothing + consensus logic."""
        per_frame = []

        for idx, items in sorted(detections.items()):
            positive, number = False, None
            for _, text, conf in items:
                ok, n = parse_boulder_number(text or "")
                if ok:
                    positive = True
                    if n is not None and number is None:
                        number = n
            per_frame.append({"idx": idx, "positive": positive, "number": number})

        starts = []
        W = self.smooth_window
        last_idx = -1e9
        prev_num = None

        for k in range(len(per_frame)):
            window = per_frame[k : k + W]
            if not window:
                break

            positives = [r for r in window if r["positive"]]
            if len(positives) / len(window) < self.majority_ratio:
                continue

            nums = [r["number"] for r in positives if r["number"] is not None]
            num = Counter(nums).most_common(1)[0][0] if nums else None

            if self.require_number and num is None:
                continue

            idx0 = window[0]["idx"]
            if idx0 - last_idx < W:
                continue

            if num != prev_num:
                starts.append({"frame_idx": idx0, "boulder": num})
                last_idx = idx0
                prev_num = num

        return starts
