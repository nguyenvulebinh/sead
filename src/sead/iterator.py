"""
Streaming iterator for SEAD.

Usage:
    iterator = SEADIterator(detector)
    for chunk in audio_chunks:
        segments = iterator(chunk)
        for s in segments:
            print(s)
    segments = iterator.flush()
"""

from pathlib import Path

import numpy as np

from sead.config import PATCH_HOP_SEC, SAMPLE_RATE
from sead.detector import SEADDetector


class SEADIterator:
    """
    Streaming iterator for sound event detection.

    feed audio chunks, get completed
    segments as they are detected. Tracks frame index internally.

    Parameters
    ----------
    detector : SEADDetector
        Preloaded SEAD detector (or path to model zip)
    sampling_rate : int (default 16000)
        Audio sample rate
    window_seconds : float (default 0.98)
        Window length per chunk (seconds)
    hop_seconds : float (default 0.48)
        Hop between chunks (seconds)
    incremental : bool (default True)
        If True, return start/end events like VADIterator (better for streaming).
        If False, return full segments when event ends.
    """

    def __init__(
        self,
        detector: SEADDetector | Path,
        *,
        sampling_rate: int = SAMPLE_RATE,
        window_seconds: float = 0.98,
        hop_seconds: float = PATCH_HOP_SEC,
        incremental: bool = True,
    ) -> None:
        if isinstance(detector, Path):
            self._detector = SEADDetector(detector)
        else:
            self._detector = detector
        self.sampling_rate = sampling_rate
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.incremental = incremental
        self.window_samples = int(round(window_seconds * sampling_rate))
        self.hop_samples = int(round(hop_seconds * sampling_rate))
        self.reset_states()

    def reset_states(self) -> None:
        """Reset internal state for a new stream."""
        self._detector.reset_stream()
        self._frame_idx = 0
        self._current_sample = 0

    def __call__(
        self,
        x: np.ndarray,
        return_seconds: bool = True,
    ) -> list:
        """
        Process one audio chunk.

        Parameters
        ----------
        x : np.ndarray
            Audio chunk, shape (n_samples,) or (n_samples, 1).
            Float32 in [-1, 1]. Length should be window_samples (0.98s at 16kHz).
        return_seconds : bool (default True)
            Segments use seconds; if False, use sample indices (not implemented)

        Returns
        -------
        If incremental=False: list of Segment (completed segments)
        If incremental=True: list of {'start': t, 'label': str} or
            {'end': t, 'label': str, 'confidence': float} (VADIterator-style)
        """
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.reshape(-1)
        x = x.astype(np.float32)

        if x.shape[0] < self.window_samples:
            pad = np.zeros(self.window_samples - x.shape[0], dtype=np.float32)
            x = np.concatenate([x, pad])

        if self.incremental:
            result = self._detector.process_stream_events(
                x, start_frame=self._frame_idx
            )
        else:
            result = self._detector.process_stream(x, start_frame=self._frame_idx)

        self._frame_idx += 1
        self._current_sample += self.hop_samples

        return result

    def flush(self, return_seconds: bool = True) -> list:
        """
        Flush any active segment at end of stream. Call after last __call__.

        Returns
        -------
        If incremental=False: list of Segment
        If incremental=True: list of {'end': t, 'label': str, 'confidence': float}
        """
        end_time = self._frame_idx * self.hop_seconds
        if self.incremental:
            return self._detector.flush_stream_events(end_time)
        return self._detector.flush_stream(end_time)
