from typing import List, Tuple

import numpy as np

from src.bouldering.scoring.utils import clamp, sigmoid


def rms_delta_score(
    delta_rms: float,
    ref: float = 0.02,
) -> float:
    """Compute a normalized score from the RMS delta.

    This score measures sudden increases in audio energy, which often
    correspond to crowd reactions such as applause or cheering.
    Only positive RMS deltas contribute to the score.

    Args:
        delta_rms (float): Difference between current RMS and a local
            baseline RMS value.
        ref (float, optional): Reference RMS delta corresponding to a
            maximal score. Defaults to 0.02.

    Returns:
        float: Normalized RMS delta score in the range [0, 1].
    """
    if delta_rms is None or delta_rms <= 0:
        return 0.0
    return clamp(delta_rms / ref)


def z_rms_score(
    z_rms: float,
    z_ref: float = 1.5,
    tau: float = 1.0,
) -> float:
    """Compute a score based on the RMS z-score.

    This score captures how loud the current audio frame is compared
    to a local temporal baseline. It is robust to continuous commentary
    and emphasizes relative loudness peaks.

    Args:
        z_rms (float): Z-score of the RMS value.
        z_ref (float, optional): Z-score threshold above which the
            score starts increasing significantly. Defaults to 1.5.
        tau (float, optional): Temperature parameter controlling the
            slope of the sigmoid. Defaults to 1.0.

    Returns:
        float: Normalized z-RMS score in the range [0, 1].
    """
    if z_rms is None:
        return 0.0
    return sigmoid((z_rms - z_ref) / tau)


def yamnet_score(
    crowd_prob: float,
) -> float:
    """Compute a weak semantic crowd score from YAMNet predictions.

    This score aggregates crowd-related semantic probabilities
    (e.g., crowd noise, cheering, shouting). It is intentionally
    weak and non-dominant, acting as a semantic hint rather than
    a primary detection signal.

    Args:
        crowd_prob (float): Summed probability of crowd-related
            YAMNet classes.

    Returns:
        float: Normalized crowd semantic score in the range [0, 1].
    """
    if crowd_prob is None or crowd_prob <= 0:
        return 0.0
    return clamp(np.sqrt(crowd_prob) * 10.0)


def audio_score_at_t(
    delta_rms: float,
    z_rms: float,
    crowd_prob: float,
    weights: Tuple[float, float, float] = (0.55, 0.30, 0.15),
) -> float:
    """Compute the audio score at a single time instant.

    The audio score is a weighted combination of:
    - RMS delta (primary signal for reactions),
    - RMS z-score (relative loudness),
    - YAMNet crowd semantic score (weak semantic hint).

    Args:
        delta_rms (float): RMS delta value.
        z_rms (float): RMS z-score.
        crowd_prob (float): Crowd semantic probability.
        weights (Tuple[float, float, float], optional): Weights for
            (delta RMS, z-RMS, YAMNet crowd). Defaults to (0.55, 0.30, 0.15).

    Returns:
        float: Audio score in the range [0, 1].
    """
    w1, w2, w3 = weights

    s1 = rms_delta_score(delta_rms)
    s2 = z_rms_score(z_rms)
    s3 = yamnet_score(crowd_prob)

    score = w1 * s1 + w2 * s2 + w3 * s3

    return clamp(score)


def compute_audio_score(
    delta_rms_signal: List[Tuple[float, float]],
    z_rms_signal: List[Tuple[float, float]],
    crowd_signal: List[Tuple[float, float]],
    weights: Tuple[float, float, float] = (0.55, 0.30, 0.15),
) -> List[Tuple[float, float]]:
    """Compute a continuous audio score over time.

    All input signals must be aligned on the same time base
    (typically interpolated onto the visual timeline).

    Args:
        delta_rms_signal (List[Tuple[float, float]]): Time series of
            RMS delta values as (time, value).
        z_rms_signal (List[Tuple[float, float]]): Time series of RMS
            z-scores as (time, value).
        crowd_signal (List[Tuple[float, float]]): Time series of
            crowd semantic probabilities as (time, value).
        weights (Tuple[float, float, float], optional): Weights for
            (delta RMS, z-RMS, YAMNet crowd). Defaults to (0.55, 0.30, 0.15).

    Returns:
        List[Tuple[float, float]]: Audio score time series as
        (time, score).
    """
    assert len(delta_rms_signal) == len(z_rms_signal) == len(crowd_signal)

    scores = []

    for (t1, dr), (_, zr), (_, cp) in zip(
        delta_rms_signal,
        z_rms_signal,
        crowd_signal,
    ):
        score = audio_score_at_t(
            delta_rms=dr,
            z_rms=zr,
            crowd_prob=cp,
            weights=weights,
        )
        scores.append((t1, score))

    return scores
