"""YAMNet audio event classifier.

Uses pretrained YAMNet from TensorFlow Hub to classify
audio segments into semantic sound classes (applause, cheering, shout, etc.).
"""

import csv
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class YamNetClassifier:
    """Wrapper around pretrained YAMNet."""

    def __init__(self):
        """Initialize YamNetClassifier."""
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")

        # load class names from the model
        class_map_path = self.model.class_map_path().numpy()
        self.class_names = self._class_names_from_csv(class_map_path)

    @staticmethod
    def _class_names_from_csv(class_map_csv_path: str) -> List[str]:
        """Returns list of class names corresponding to YAMNet score vector.

        Args:
            class_map_csv_path: Path to YAMNet class map CSV file.

        Returns:
            List of class display names in model output order.
        """
        class_names = []
        with tf.io.gfile.GFile(class_map_csv_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row["display_name"])
        return class_names

    def predict(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[float, Dict[str, float]]]:
        """Run YAMNet on an audio waveform.

        Args:
            waveform: mono float32 waveform [-1, 1]
            sample_rate: sampling rate (must be 16 kHz)

        Returns:
            List of (time, {class_name: probability})
        """
        if sample_rate != 16000:
            raise ValueError("YAMNet requires 16 kHz audio")

        scores, embeddings, spectrogram = self.model(waveform)
        scores = scores.numpy()

        frame_duration = 0.48  # seconds per YAMNet frame
        results = []

        for i, frame_scores in enumerate(scores):
            t = i * frame_duration
            results.append(
                (
                    t,
                    {self.class_names[j]: float(frame_scores[j]) for j in range(len(self.class_names))},
                )
            )

        return results
