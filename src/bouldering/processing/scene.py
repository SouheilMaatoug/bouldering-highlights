# src/bouldering/scenes_ocr.py
# Minimal pipeline: uses OcrService + smoothing to return start indices of each "Boulder N".
from collections import Counter
from typing import List

from src.bouldering.processing.ocr import OCRDetector
from src.bouldering.processing.parser import parse_boulder_number


class SceneSplitterOCR:
    """
    Minimal "scene splitter" driven by OCR 'Boulder N' overlays.
    Returns frame indices of the beginning of each confirmed Boulder section.
    """

    def __init__(
        self,
        langs: List,
        crop_box: List,
        fx=0.5,
        fy=0.5,
        stride=3,
        batch_size=16,
        smooth_window=5,
        majority_ratio=0.60,
        require_number=True,
    ):
        self.langs = langs or ["en"]
        self.stride = stride
        self.batch_size = batch_size
        self.crop_box = crop_box
        self.fx = fx
        self.fy = fy
        self.smooth_window = smooth_window
        self.majority_ratio = majority_ratio
        self.require_number = require_number
        self._svc = OCRDetector(langs=self.langs)

    def split(self, vid):
        """
        Returns: list of dicts: { "frame_idx": int, "boulder": Optional[int] }
        """
        # 1) OCR per sampled frame
        frames_text = self._svc.run(
            vid,
            stride=self.stride,
            batch_size=self.batch_size,
            crop_box=self.crop_box,
            fx=self.fx,
            fy=self.fy,
            min_conf=0.0,
        )

        # 2) Per-frame boolean + number
        idxs = sorted(frames_text.keys())
        per_frame = []
        for i in idxs:
            entries = frames_text.get(i, [])
            positive = False
            number = None
            for e in entries:
                ok, n = parse_boulder_number(e.get("text", ""))
                if ok:
                    positive = True
                    if n is not None and number is None:
                        number = n
            per_frame.append({"idx": i, "positive": positive, "number": number})

        # 3) Smoothing with lookahead majority
        starts = []
        W = max(1, int(self.smooth_window))
        last_confirm_idx = -(10**9)
        prev_num = None

        k = 0
        while k < len(per_frame):
            window = per_frame[k : k + W]
            if not window:
                break

            pos_count = sum(1 for r in window if r["positive"])
            ratio = pos_count / len(window)

            consensus_positive = ratio >= self.majority_ratio
            consensus_num = None
            if consensus_positive:
                nums = [r["number"] for r in window if r["positive"] and r["number"] is not None]
                if nums:
                    consensus_num = Counter(nums).most_common(1)[0][0]

            if consensus_positive:
                # if requiring number, skip confirm if no number found
                if self.require_number and consensus_num is None:
                    k += 1
                    continue

                start_idx = per_frame[k]["idx"]
                far_enough = (start_idx - last_confirm_idx) >= W

                if self.require_number:
                    number_changed = consensus_num != prev_num
                    if far_enough and number_changed:
                        starts.append({"frame_idx": start_idx, "boulder": consensus_num})
                        last_confirm_idx = start_idx
                        prev_num = consensus_num
                        k += W
                        continue
                else:
                    if far_enough:
                        starts.append({"frame_idx": start_idx, "boulder": consensus_num})
                        last_confirm_idx = start_idx
                        prev_num = consensus_num
                        k += W
                        continue

            k += 1

        return starts
