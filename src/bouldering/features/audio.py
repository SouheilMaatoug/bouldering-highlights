from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, stft


def resample_waveform(
    waveform: np.ndarray,
    original_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample a mono waveform to a target sampling rate.

    Args:
        waveform: Audio waveform (mono or stereo)
        original_sr: Original sampling rate (Hz)
        target_sr: Target sampling rate (Hz)

    Returns:
        Resampled mono waveform (float32)
    """
    if original_sr == target_sr:
        return waveform.astype(np.float32)

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    gcd = np.gcd(original_sr, target_sr)
    up = target_sr // gcd
    down = original_sr // gcd

    resampled = resample_poly(waveform, up, down)
    return resampled.astype(np.float32)


def apply_filter(
    waveform: np.ndarray,
    sample_rate: int,
    filter_type: str,
    low_hz: Optional[float] = None,
    high_hz: Optional[float] = None,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth filter to a waveform.

    Args:
        waveform: Mono audio waveform
        sample_rate: Sampling rate (Hz)
        filter_type: 'bandpass', 'bandstop', 'highpass'
        low_hz: Low cutoff frequency
        high_hz: High cutoff frequency
        order: Filter order

    Returns:
        Filtered waveform
    """
    if filter_type == "bandpass":
        assert low_hz and high_hz
        Wn = [low_hz, high_hz]
    elif filter_type == "bandstop":
        assert low_hz and high_hz
        Wn = [low_hz, high_hz]
    elif filter_type == "highpass":
        assert low_hz
        Wn = low_hz
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    sos = butter(
        order,
        Wn,
        btype=filter_type,
        fs=sample_rate,
        output="sos",
    )

    return sosfilt(sos, waveform)


def rms_energy(
    waveform: np.ndarray,
    sample_rate: int,
    window_seconds: float,
) -> List[Tuple[float, float]]:
    """Compute RMS energy over sliding windows."""
    window_size = int(window_seconds * sample_rate)
    rms = []

    for i in range(0, len(waveform) - window_size, window_size):
        chunk = waveform[i : i + window_size]
        value = float(np.sqrt(np.mean(chunk**2)))
        t = i / sample_rate
        rms.append((t, value))

    return rms


def delta_signal(signal: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Compute first temporal derivative of a signal."""
    return [(signal[i][0], signal[i][1] - signal[i - 1][1]) for i in range(1, len(signal))]


def zscore_signal(
    signal: List[Tuple[float, float]],
    window: int = 10,
) -> List[Tuple[float, float]]:
    """Compute rolling z-score for a signal."""
    values = [v for _, v in signal]
    out = []

    for i, (t, v) in enumerate(signal):
        ref = values[max(0, i - window) : i]
        if len(ref) < 3:
            out.append((t, 0.0))
        else:
            mu = np.mean(ref)
            sigma = np.std(ref) + 1e-6
            out.append((t, (v - mu) / sigma))

    return out


def spectral_centroid(
    waveform: np.ndarray,
    sample_rate: int,
    window_seconds: float,
) -> List[Tuple[float, float]]:
    """Compute spectral centroid over sliding windows."""
    nperseg = int(window_seconds * sample_rate)

    f, t, Zxx = stft(
        waveform,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=0,
    )

    centroids = []
    for i in range(Zxx.shape[1]):
        mag = np.abs(Zxx[:, i])
        if mag.sum() < 1e-6:
            centroids.append((t[i], 0.0))
        else:
            c = float(np.sum(f * mag) / np.sum(mag))
            centroids.append((t[i], c))

    return centroids


class AudioFeatures:
    """Audio feature extractor.

    Handles:
    - resampling
    - filtering
    - temporal features (RMS, delta RMS, z-score)
    - spectral features (centroid)
    """

    def __init__(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ):
        """Initialize the AudioFeatures class.

        Args:
            waveform: Raw audio waveform (mono or stereo)
            sample_rate: Sampling rate (Hz)
        """
        self.raw_waveform = waveform
        self.sample_rate = sample_rate

        self.waveform = waveform
        self.filtered_waveform: Optional[np.ndarray] = None

    def resample(self, target_sr: int):
        """Resample waveform to target sampling rate."""
        self.waveform = resample_waveform(
            self.waveform,
            self.sample_rate,
            target_sr,
        )
        self.sample_rate = target_sr
        return self

    def filter(
        self,
        filter_type: str,
        low_hz: Optional[float] = None,
        high_hz: Optional[float] = None,
    ):
        """Apply a filter to the waveform."""
        self.filtered_waveform = apply_filter(
            self.waveform,
            self.sample_rate,
            filter_type=filter_type,
            low_hz=low_hz,
            high_hz=high_hz,
        )
        return self

    def compute_features(
        self,
        window_seconds: float = 0.5,
    ):
        """Compute audio features from the (filtered) waveform.

        Returns:
            Dict with RMS, delta RMS, z-score RMS, spectral centroid
        """
        signal = self.filtered_waveform if self.filtered_waveform is not None else self.waveform

        rms = rms_energy(signal, self.sample_rate, window_seconds)
        delta_rms = delta_signal(rms)
        z_rms = zscore_signal(rms)
        centroid = spectral_centroid(signal, self.sample_rate, window_seconds)
        delta_centroid = delta_signal(centroid)

        return {
            "rms": rms,
            "delta_rms": delta_rms,
            "z_rms": z_rms,
            "spectral_centroid": centroid,
            "delta_spectral_centroid": delta_centroid,
        }
