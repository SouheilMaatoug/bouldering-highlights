def compute_overall_score(audio_score, visual_score, alpha=0.65, beta=0.35):
    """Compute the overall multimodal score from visual and audio scores.

    The overall score is a weighted combination of the visual score and
    the audio score, computed at each time instant. Both input signals
    must be aligned on the same time base.

    Args:
        audio_score (list[tuple[float, float]]): Audio score time series
            as (time, score) pairs.
        visual_score (list[tuple[float, float]]): Visual score time series
            as (time, score) pairs.
        alpha (float, optional): Weight for the visual score contribution.
            Defaults to 0.65.
        beta (float, optional): Weight for the audio score contribution.
            Defaults to 0.35.

    Returns:
        list[tuple[float, float]]: Overall multimodal score as a time
        series of (time, score) pairs, with scores clamped to [0, 1].

    Raises:
        AssertionError: If the input score series are not of the same length.
    """
    overall_score = []

    for (t1, v), (_, a) in zip(visual_score, audio_score):
        s = alpha * v + beta * a
        overall_score.append((t1, min(1.0, s)))

    return overall_score
