import numpy as np
import soundfile as sf


class Audio:
    """Class for defining an audio track."""

    def __init__(self, samples: np.ndarray, sample_rate: int) -> None:
        """Initialize the Audio class.

        Args:
            samples (np.ndarray): The array of samples.
            sample_rate (int): The sample rate.
        """
        self._samples = samples
        self._sample_rate = sample_rate

    @classmethod
    def read(cls, path: str) -> "Audio":
        """A class method for reading an audio file.

        Args:
            path (str): The path to the audio file.

        Returns:
            Audio: An Audio instance.
        """
        samples, sr = sf.read(path)
        return cls(samples, sr)

    @property
    def samples(self):
        return self._samples

    @property
    def sample_rate(self):
        return self._sample_rate

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

    def write(self, output_path: str) -> None:
        """Write to an audio file.

        Args:
            output_path (str): The path to the audio file.
        """
        sf.write(output_path, self.samples, self.sample_rate)
