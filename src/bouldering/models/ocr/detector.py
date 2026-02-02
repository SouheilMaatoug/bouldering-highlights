from typing import List, Optional

import easyocr
import numpy as np
import torch


class OCRDetector:
    """Optical Character Recognition using EasyOCR."""

    def __init__(
        self,
        langs: list[str] = ["en"],
        use_gpu: Optional[bool] = None,
        decoder: str = "greedy",
    ):
        """Initalize the OCR detector.

        Args:
            langs (list[str], optional): Language codes to be recognized. Defaults to ["en"].
            use_gpu (Optional[bool], optional): Whether to use GPU acceleration. Defaults to None.
            decoder (str, optional): Decoding algorithm to use (`greedy`, `beamsearch`). Defaults to "greedy".
        """
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(langs, gpu=use_gpu)
        self.decoder = decoder

    def read_batch(self, images: List[np.ndarray]) -> List[list]:
        """Run OCR on a batch of images.

        Args:
            images (List[np.ndarray]): List of imagtes to process.

        Returns:
            List[list]: A list of detection per image.
        """
        return self.reader.readtext_batched(images, detail=1, decoder=self.decoder)
