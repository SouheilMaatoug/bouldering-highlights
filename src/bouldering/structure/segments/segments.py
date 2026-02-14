from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    """Temporal segment in seconds."""

    start: float
    end: float
    score: Optional[float] = None
    label: Optional[str] = None

    @property
    def duration(self) -> float:
        """Segment duration."""
        return max(0.0, self.end - self.start)


def segments_from_peaks(
    peaks: list[dict],
    pre: float = 1.5,
    post: float = 2.5,
) -> list[Segment]:
    """Build segments around detected peaks."""
    segments = []

    for p in peaks:
        segments.append(
            Segment(
                start=max(0.0, p["time"] - pre),
                end=p["time"] + post,
                score=p.get("score"),
                label="PEAK",
            )
        )

    return segments


def segments_from_events(
    events: list[dict],
    pre: float = 0.5,
    post: float = 0.7,
) -> list[Segment]:
    """Build segments around detected events."""
    segments = []

    for e in events:
        segments.append(
            Segment(
                start=max(0.0, e["start"] - pre),
                end=e["end"] + post,
                score=e.get("confidence"),
                label=e["type"],
            )
        )

    return segments


def merge_segments(segments):
    """Merge overlapping time segments."""
    if not segments:
        return []

    segments = sorted(segments, key=lambda s: s["start"])
    merged = [segments[0]]

    for s in segments[1:]:
        last = merged[-1]
        if s["start"] <= last["end"]:
            last["end"] = max(last["end"], s["end"])
            last["peak_score"] = max(last["peak_score"], s["peak_score"])
        else:
            merged.append(s)

    return merged


def rank_segments(segments, duration_weight=0.1):
    """Rank highlight segments.

    Returns:
        Sorted list (best first)
    """
    ranked = []

    for s in segments:
        duration = s["end"] - s["start"]
        score = s["peak_score"] + duration_weight * duration
        ranked.append(
            {
                **s,
                "duration": duration,
                "rank_score": score,
            }
        )

    return sorted(ranked, key=lambda x: x["rank_score"], reverse=True)
