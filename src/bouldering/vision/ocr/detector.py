from typing import List, Optional

import easyocr
import numpy as np
import torch


class OCRDetector:
    """Low-level OCR inference wrapper."""

    def __init__(
        self,
        langs: list[str] = ["en"],
        use_gpu: Optional[bool] = None,
        decoder: str = "greedy",
    ):
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(langs, gpu=use_gpu)
        self.decoder = decoder

    def read_batch(self, images: List[np.ndarray]) -> List[list]:
        """Run OCR on a batch of images."""
        return self.reader.readtext_batched(images, detail=1, decoder=self.decoder)
