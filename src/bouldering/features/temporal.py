"""Temporal aggregation utilities."""

from typing import List, Tuple


def sliding_window(
    values: List[Tuple[float, float]],
    window_seconds: float,
) -> List[List[Tuple[float, float]]]:
    """Split a signal into sliding time windows."""
    windows = []

    for i, (t, _) in enumerate(values):
        window = [(t2, v2) for (t2, v2) in values if t2 >= t - window_seconds and t2 <= t]
        windows.append(window)

    return windows


def aggregate_mean(window: List[Tuple[float, float]]) -> float:
    """Compute mean aggregation."""
    return sum(v for _, v in window) / len(window)


def aggregate_max(window: List[Tuple[float, float]]) -> float:
    """Compute max aggregation."""
    return max(v for _, v in window)


def displacement_amplitude(
    window: list[Tuple[float, float]],
) -> float:
    """Compute displacement amplitude over a time window."""
    values = [v for _, v in window]
    return max(values) - min(values)
