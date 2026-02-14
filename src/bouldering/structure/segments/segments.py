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


def merge_segments(
    segments: list[Segment],
    gap: float = 0.2,
) -> list[Segment]:
    """Merge overlapping or close temporal segments."""
    if not segments:
        return []

    segments = sorted(segments, key=lambda s: s.start)
    merged = [segments[0]]

    for s in segments[1:]:
        last = merged[-1]

        if s.start <= last.end + gap:
            last.end = max(last.end, s.end)

            # propagate strongest score if present
            if s.score is not None:
                if last.score is None:
                    last.score = s.score
                else:
                    last.score = max(last.score, s.score)

            # merge labels if needed
            if s.label and s.label not in (last.label or ""):
                last.label = f"{last.label}+{s.label}" if last.label else s.label
        else:
            merged.append(s)

    return merged


def rank_segments(
    segments: list[Segment],
    duration_weight: float = 0.1,
) -> list[Segment]:
    """Rank segments by score and duration."""

    def rank_value(s: Segment) -> float:
        base = s.score or 0.0
        return base + duration_weight * s.duration

    return sorted(segments, key=rank_value, reverse=True)
