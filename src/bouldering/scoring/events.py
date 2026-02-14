# src/bouldering/scoring/events_boost.py

from typing import Dict, List

import numpy as np


def event_boost(
    t: float,
    event_time: float,
    confidence: float,
    tau: float,
    max_boost: float,
) -> float:
    """Compute temporal score boost induced by a single event.

    The boost decays exponentially with temporal distance to the event.

    Args:
        t (float): Current timestamp.
        event_time (float): Central time of the event.
        confidence (float): Event confidence in [0, 1].
        tau (float): Temporal decay constant (seconds).
        max_boost (float): Maximum boost amplitude.

    Returns:
        float: Boost value.
    """
    return max_boost * confidence * np.exp(-abs(t - event_time) / tau)


def apply_event_boosts(
    base_score: List[tuple[float, float]],
    events: List[dict],
    event_params: Dict[str, dict],
) -> List[tuple[float, float]]:
    """Apply event-based temporal boosts to a score curve.

    Args:
        base_score: [(time, score)] base score curve.
        events: List of detected events.
        event_params: Per-event-type boost parameters.

    Returns:
        Boosted score curve [(time, score)].
    """
    boosted = []

    for t, s in base_score:
        boost = 0.0

        for e in events:
            if e["type"] not in event_params:
                continue

            params = event_params[e["type"]]
            t_event = 0.5 * (e["start"] + e["end"])

            boost += event_boost(
                t=t,
                event_time=t_event,
                confidence=e.get("confidence", 0.5),
                tau=params["tau"],
                max_boost=params["max_boost"],
            )

        boosted.append((t, min(1.0, s * (1.0 + boost))))

    return boosted
