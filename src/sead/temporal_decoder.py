"""Hysteresis-based temporal decoding for event onset/offset detection."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from sead.config import (
    OFFSET_THRESHOLD,
    ONSET_THRESHOLD,
    PATCH_HOP_SEC,
    TARGET_LABELS,
)


@dataclass
class RawSegment:
    """Raw segment from temporal decoder (before merge/filter)."""

    start_time: float
    end_time: float
    label: str
    confidence: float


def _event_tuple(
    event_type: str,
    t: float,
    label: str,
    confidence: float | None,
    start_time: float | None = None,
) -> tuple[str, float, str, float | None, float | None]:
    """(event_type, time, label, confidence, start_time). start_time only for 'end'."""
    return (event_type, t, label, confidence, start_time)


class TemporalDecoder:
    """Hysteresis state machine: onset when above upper threshold, offset when below lower."""

    def __init__(
        self,
        onset_threshold: float = ONSET_THRESHOLD,
        offset_threshold: float = OFFSET_THRESHOLD,
        frame_hop_sec: float = PATCH_HOP_SEC,
    ) -> None:
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_hop_sec = frame_hop_sec
        self._active_label: str | None = None
        self._start_time: float = 0.0
        self._max_confidence: float = 0.0

    def _decode_yield(
        self, probs: np.ndarray, start_frame: int = 0
    ) -> Iterator[tuple[str, float, str, float | None, float | None]]:
        """
        Single hysteresis loop. Yields (event_type, time, label, confidence, start_time).
        event_type: 'start' | 'end'. start_time only for 'end'.
        """
        labels = TARGET_LABELS
        for i in range(probs.shape[0]):
            t = (start_frame + i) * self.frame_hop_sec
            idx = int(np.argmax(probs[i]))
            label = labels[idx]
            conf = float(probs[i, idx])

            if self._active_label is None:
                if conf >= self.onset_threshold:
                    self._active_label = label
                    self._start_time = t
                    self._max_confidence = conf
                    yield _event_tuple("start", t, label, None, None)
            else:
                if label == self._active_label:
                    if conf < self.offset_threshold:
                        yield _event_tuple(
                            "end",
                            t,
                            self._active_label,
                            self._max_confidence,
                            self._start_time,
                        )
                        self._active_label = None
                    else:
                        self._max_confidence = max(
                            self._max_confidence, conf
                        )
                else:
                    if conf >= self.onset_threshold:
                        yield _event_tuple(
                            "end",
                            t,
                            self._active_label,
                            self._max_confidence,
                            self._start_time,
                        )
                        self._active_label = label
                        self._start_time = t
                        self._max_confidence = conf
                        yield _event_tuple("start", t, label, None, None)

    def decode(self, probs: np.ndarray, start_frame: int = 0) -> list[RawSegment]:
        """
        Decode frame-wise probs into segments using hysteresis.

        Args:
            probs: [N, 3] smoothed target probs (speech, music, others)
            start_frame: frame index offset for time calculation

        Returns:
            List of raw segments
        """
        segments: list[RawSegment] = []
        for evt_type, t, label, conf, start_t in self._decode_yield(
            probs, start_frame
        ):
            if evt_type == "end" and start_t is not None:
                segments.append(RawSegment(start_t, t, label, conf or 0.0))
        return segments

    def decode_events(
        self, probs: np.ndarray, start_frame: int = 0
    ) -> list[dict]:
        """
        Decode frame-wise probs into incremental start/end events (VADIterator-style).

        Returns:
            List of {'start': t, 'label': str} or {'end': t, 'label': str, 'confidence': float}
        """
        events: list[dict] = []
        for evt_type, t, label, conf, _ in self._decode_yield(
            probs, start_frame
        ):
            if evt_type == "start":
                events.append({"start": round(t, 4), "label": label})
            else:
                events.append({"end": round(t, 4), "label": label, "confidence": round(conf or 0.0, 4)})
        return events

    def flush_events(self, end_time: float) -> list[dict]:
        """Flush any active segment as end event."""
        events: list[dict] = []
        if self._active_label is not None:
            events.append(
                {
                    "end": round(end_time, 4),
                    "label": self._active_label,
                    "confidence": round(self._max_confidence, 4),
                }
            )
            self._active_label = None
        return events

    def flush(self, end_time: float) -> list[RawSegment]:
        """Flush any active segment (e.g. at end of stream)."""
        segments: list[RawSegment] = []
        if self._active_label is not None:
            segments.append(
                RawSegment(
                    self._start_time,
                    end_time,
                    self._active_label,
                    self._max_confidence,
                )
            )
            self._active_label = None
        return segments

    def reset(self) -> None:
        """Reset state for new stream."""
        self._active_label = None
