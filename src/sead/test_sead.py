"""Tests and debug script for SEAD."""

import numpy as np

from sead.class_mapping import aggregate_to_target_classes
from sead.segment_builder import build_segments, merge_nearby_segments
from sead.smoothing import EMASmoother
from sead.temporal_decoder import RawSegment, TemporalDecoder


def test_class_mapping() -> None:
    """Class aggregation sums correctly."""
    logits = np.zeros((2, 521), dtype=np.float32)
    logits[0, 0] = 5.0  # speech index -> high prob
    logits[0, 24] = 3.0  # music index
    out = aggregate_to_target_classes(logits)
    assert out.shape == (2, 3)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_ema_smoother() -> None:
    """EMA reduces variance."""
    smoother = EMASmoother(alpha=0.4, num_classes=3)
    p1 = np.array([0.9, 0.05, 0.05], dtype=np.float32)
    p2 = np.array([0.1, 0.8, 0.1], dtype=np.float32)
    s1 = smoother.update(p1)
    s2 = smoother.update(p2)
    assert np.allclose(s1, p1)
    assert 0.1 < s2[0] < 0.9
    assert 0.05 < s2[1] < 0.8


def test_temporal_decoder() -> None:
    """Hysteresis produces segments."""
    decoder = TemporalDecoder(onset_threshold=0.5, offset_threshold=0.35)
    probs = np.array(
        [
            [0.2, 0.2, 0.6],
            [0.7, 0.15, 0.15],
            [0.7, 0.15, 0.15],
            [0.3, 0.2, 0.5],
        ],
        dtype=np.float32,
    )
    raw = decoder.decode(probs)
    raw.extend(decoder.flush(4.0 * 0.48))
    assert len(raw) >= 1
    labels = [s.label for s in raw]
    assert "speech" in labels or "others" in labels


def test_merge_segments() -> None:
    """Nearby segments merge."""
    raw = [
        RawSegment(0.0, 1.0, "speech", 0.8),
        RawSegment(1.2, 2.0, "speech", 0.7),
        RawSegment(3.0, 4.0, "speech", 0.9),
    ]
    merged = merge_nearby_segments(raw, max_gap_sec=0.5)
    assert len(merged) == 2
    assert merged[0].end_time == 2.0
    assert merged[1].start_time == 3.0


def test_build_segments() -> None:
    """Full pipeline: merge + duration filter."""
    raw = [
        RawSegment(0.0, 0.3, "speech", 0.8),
        RawSegment(0.5, 2.0, "speech", 0.7),
    ]
    out = build_segments(raw, min_duration_sec=0.5, max_gap_sec=0.3)
    assert len(out) == 1
    assert out[0].start_time == 0.0
    assert out[0].end_time == 2.0
    assert out[0].label == "speech"


def run() -> None:
    """Run all tests."""
    test_class_mapping()
    test_ema_smoother()
    test_temporal_decoder()
    test_merge_segments()
    test_build_segments()
    print("All tests passed.")


if __name__ == "__main__":
    run()
