import numpy as np


class Audio:
    """Class for defining an audio track."""

    def __init__(self, samples: np.ndarray, sample_rate: int) -> None:
        """Initialize the Audio class.

        Args:
            samples (np.ndarray): The array of samples.
            sample_rate (int): The sample rate.
        """
        self.samples = samples
        self.sample_rate = sample_rate

    @property
    def duration(self) -> float:
        """Get the duration of the audio track.

        Returns:
            float: The duration in seconds.
        """
        return len(self.samples) / self.sample_rate

    def cut(self, start_time: float, end_time: float) -> "Audio":
        """Cut an audio track between two timestamps.

        Args:
            start_time (float): The start time in seconds.
            end_time (float): The end time in seconds.

        Returns:
            Audio: The new Audio instance.
        """
        start = int(start_time * self.sample_rate)
        end = int(end_time * self.sample_rate)
        return Audio(self.samples[start:end], self.sample_rate)
