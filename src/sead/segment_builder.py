"""Gap merging and duration filtering for segments."""

from sead.config import MAX_GAP_SEC, MIN_DURATION_SEC, Segment
from sead.temporal_decoder import RawSegment


def merge_nearby_segments(
    segments: list[RawSegment],
    max_gap_sec: float = MAX_GAP_SEC,
) -> list[RawSegment]:
    """Merge segments of same label if gap between them <= max_gap_sec."""
    if not segments:
        return []

    merged: list[RawSegment] = [segments[0]]
    for s in segments[1:]:
        last = merged[-1]
        if (
            s.label == last.label
            and (s.start_time - last.end_time) <= max_gap_sec
        ):
            merged[-1] = RawSegment(
                last.start_time,
                s.end_time,
                last.label,
                max(last.confidence, s.confidence),
            )
        else:
            merged.append(s)
    return merged


def filter_by_duration(
    segments: list[RawSegment],
    min_duration_sec: float = MIN_DURATION_SEC,
) -> list[RawSegment]:
    """Drop segments shorter than min_duration_sec."""
    return [s for s in segments if (s.end_time - s.start_time) >= min_duration_sec]


def build_segments(
    raw: list[RawSegment],
    min_duration_sec: float = MIN_DURATION_SEC,
    max_gap_sec: float = MAX_GAP_SEC,
) -> list[Segment]:
    """Merge nearby segments and filter by duration. Returns final Segment list."""
    merged = merge_nearby_segments(raw, max_gap_sec)
    filtered = filter_by_duration(merged, min_duration_sec)
    return [
        Segment(s.start_time, s.end_time, s.label, s.confidence)
        for s in filtered
    ]
