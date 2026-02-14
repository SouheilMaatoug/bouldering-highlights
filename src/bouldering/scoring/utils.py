import numpy as np


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to a specified range.

    Args:
        x (float): Input value.
        lo (float, optional): Lower bound. Defaults to 0.0.
        hi (float, optional): Upper bound. Defaults to 1.0.

    Returns:
        float: Value clamped to the interval [lo, hi].
    """
    return max(lo, min(hi, x))


def interpolate_signal(
    source: list[tuple[float, float]],
    target_times: list[float],
) -> list[tuple[float, float]]:
    """Interpolate a time series onto a new time base.

    This function resamples a signal defined on a sparse or irregular
    time grid onto a new set of target timestamps using linear
    interpolation. Values outside the source time range are extrapolated
    using the boundary values.

    Args:
        source (list[tuple[float, float]]): Source signal as a list of
            (time, value) pairs.
        target_times (list[float]): Target timestamps onto which the
            signal is interpolated.

    Returns:
        list[tuple[float, float]]: Interpolated signal as a list of
        (time, interpolated_value) pairs.
    """
    src_times = np.array([t for t, v in source])
    src_vals = np.array([v for t, v in source])

    tgt_vals = np.interp(
        target_times,
        src_times,
        src_vals,
        left=src_vals[0],
        right=src_vals[-1],
    )

    return list(zip(target_times, tgt_vals))


def fill_none_with_zero(signal: list[tuple[float, float | None]]) -> list[tuple[float, float]]:
    """Replace missing values in a time series with zeros.

    This utility is typically used to clean feature signals before
    scoring, ensuring that undefined values (None) do not propagate
    into numerical computations.

    Args:
        signal (list[tuple[float, float | None]]): Input signal as
            (time, value) pairs, where value may be None.

    Returns:
        list[tuple[float, float]]: Signal with None values replaced
        by 0.0.
    """
    return [(t, v if v is not None else 0.0) for t, v in signal]


def sigmoid(x: float) -> float:
    """Compute the logistic sigmoid function.

    The sigmoid function maps real-valued inputs to the range (0, 1)
    and is commonly used to smoothly threshold values.

    Args:
        x (float): Input value.

    Returns:
        float: Sigmoid of the input.
    """
    return 1.0 / (1.0 + np.exp(-x))
