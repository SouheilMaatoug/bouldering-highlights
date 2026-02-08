"""Motion features computed from time-aware pose signals."""

from typing import List, Optional, Tuple


def time_derivative(values: List[Tuple[float, Optional[float]]]) -> List[Tuple[float, Optional[float]]]:
    """Compute time-normalized derivative of a signal.

    Args:
        values: List of (time, value).

    Returns:
        List of (time, derivative) aligned with input.
    """
    derivatives = [(values[0][0], None)]

    for i in range(1, len(values)):
        t1, v1 = values[i - 1]
        t2, v2 = values[i]

        if v1 is None or v2 is None:
            derivatives.append((t2, None))
            continue

        dt = t2 - t1
        if dt <= 0:
            derivatives.append((t2, None))
            continue

        derivatives.append((t2, (v2 - v1) / dt))

    return derivatives
