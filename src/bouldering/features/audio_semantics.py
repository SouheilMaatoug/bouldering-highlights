from typing import Dict, List, Tuple


def crowd_prediction(
    yamnet_output: List[Tuple[float, Dict[str, float]]],
    target_classes=("Crowd", "Cheering", "Shout"),
) -> List[Tuple[float, float]]:
    """Aggregate YAMNet probabilities for crowd-related classes.

    Args:
        yamnet_output: Output of YamNetClassifier.predict
        target_classes: Class names to aggregate

    Returns:
        List of (time, summed_probability)
    """
    out = []
    for t, scores in yamnet_output:
        p = sum(scores.get(cls, 0.0) for cls in target_classes)
        out.append((t, p))
    return out
