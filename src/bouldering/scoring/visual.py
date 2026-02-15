from typing import List, Tuple

from src.bouldering.scoring.utils import clamp


def upward_score(
    v_y: float,
    v_ref: float = 0.5,
) -> float:
    """Score upward vertical motion.

    Args:
        v_y: Normalized vertical velocity (upward > 0)
        v_ref: Reference velocity for saturation

    Returns:
        Score in [0, 1]
    """
    if v_y <= 0:
        return 0.0
    return clamp(v_y / v_ref)


def dynamic_score(
    v_y: float,
    v_ref: float = 0.5,
) -> float:
    """Score dynamic vertical motion (absolute velocity).

    Captures dynos and falls as visually intense motion.
    """
    return clamp(abs(v_y) / v_ref)


def effort_score(
    displacement: float,
    upward: float,
    disp_ref: float = 0.3,
) -> float:
    """Score sustained effort with limited upward progress.

    Args:
        displacement: Normalized displacement amplitude
        upward: Upward score in [0, 1]
        disp_ref: Reference displacement for saturation

    Returns:
        Score in [0, 1]
    """
    base = clamp(displacement / disp_ref)
    return base * (1.0 - upward)


def visual_score_at_t(
    v_y: float,
    displacement: float,
    weights: Tuple[float, float, float] = (0.45, 0.35, 0.20),
    v_ref: float = 0.5,
    disp_ref: float = 0.3,
) -> float:
    """Compute visual score at a single time instant.

    Args:
        v_y: Normalized vertical velocity
        displacement: Normalized displacement amplitude
        weights: (w_up, w_dyn, w_effort)
        v_ref: Reference vertical velocity
        disp_ref: Reference displacement

    Returns:
        Visual score in [0, 1]
    """
    w_up, w_dyn, w_eff = weights

    s_up = upward_score(v_y, v_ref)
    s_dyn = dynamic_score(v_y, v_ref)
    s_eff = effort_score(displacement, s_up, disp_ref)

    score = w_up * s_up + w_dyn * s_dyn + w_eff * s_eff

    return clamp(score)


def compute_visual_score(
    v_y_signal: List[Tuple[float, float]],
    displacement_signal: List[Tuple[float, float]],
    weights: Tuple[float, float, float] = (0.45, 0.35, 0.20),
    v_ref: float = 0.5,
    disp_ref: float = 0.3,
) -> List[Tuple[float, float]]:
    """Compute continuous visual score over time.

    Args:
        v_y_signal: List of (time, normalized vertical velocity)
        displacement_signal: List of (time, normalized displacement)
        weights: Component weights (upward, dynamic, effort)
        v_ref: Reference velocity for normalization
        disp_ref: Reference displacement for normalization

    Returns:
        List of (time, visual_score)
    """
    assert len(v_y_signal) == len(displacement_signal)

    scores = []

    for (t1, v_y), (t2, disp) in zip(v_y_signal, displacement_signal):
        if t1 != t2:
            raise ValueError("Time mismatch between visual signals")

        score = visual_score_at_t(
            v_y=v_y,
            displacement=disp,
            weights=weights,
            v_ref=v_ref,
            disp_ref=disp_ref,
        )

        scores.append((t1, score))

    return scores
