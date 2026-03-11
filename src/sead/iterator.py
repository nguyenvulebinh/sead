"""
Streaming iterator for SEAD.

Accepts arbitrary-sized audio chunks; buffers internally until a full window
is available. Caller can feed chunks as they arrive (e.g. from mic or network).

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

from sead.config import DEFAULT_NUM_THREADS, PATCH_HOP_SEC, SAMPLE_RATE
from sead.detector import SEADDetector


class SEADIterator:
    """
    Streaming iterator for sound event detection.

    Accepts arbitrary-sized audio chunks. Buffers internally until a full
    window (window_samples) is available, then processes and advances by hop.

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
    num_threads : int | None (default None)
        Max CPU threads for ONNX inference. If None, uses DEFAULT_NUM_THREADS.
        Only used when detector is a Path (creates SEADDetector internally).
    """

    def __init__(
        self,
        detector: SEADDetector | Path,
        *,
        sampling_rate: int = SAMPLE_RATE,
        window_seconds: float = 0.98,
        hop_seconds: float = PATCH_HOP_SEC,
        incremental: bool = True,
        num_threads: int | None = None,
    ) -> None:
        if isinstance(detector, Path):
            threads = num_threads if num_threads is not None else DEFAULT_NUM_THREADS
            self._detector = SEADDetector(detector, num_threads=threads)
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
        self._buffer = np.array([], dtype=np.float32)
        self._frames_processed = 0

    def __call__(
        self,
        x: np.ndarray,
        return_seconds: bool = True,
    ) -> list:
        """
        Process one audio chunk. Chunk may be any size; buffering is handled
        internally. Returns segments/events for all windows processed in this call.

        Parameters
        ----------
        x : np.ndarray
            Audio chunk, shape (n_samples,) or (n_samples, 1).
            Float32 in [-1, 1]. Arbitrary length.
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

        self._buffer = np.concatenate([self._buffer, x])
        all_results: list = []

        while len(self._buffer) >= self.window_samples:
            window = self._buffer[: self.window_samples].copy()
            self._buffer = self._buffer[self.hop_samples :]

            if self.incremental:
                result = self._detector.process_stream_events(
                    window, start_frame=self._frame_idx
                )
            else:
                result = self._detector.process_stream(
                    window, start_frame=self._frame_idx
                )

            all_results.extend(result)
            self._frame_idx += 1
            self._frames_processed += 1
            self._current_sample += self.hop_samples

        return all_results

    def flush(self, return_seconds: bool = True) -> list:
        """
        Flush any buffered audio (padded if needed) and active segments.
        Call after last __call__.

        Returns
        -------
        If incremental=False: list of Segment
        If incremental=True: list of {'end': t, 'label': str, 'confidence': float}
        """
        all_results: list = []

        # Process remaining buffer (pad to full window if needed)
        if len(self._buffer) > 0:
            pad_len = self.window_samples - len(self._buffer)
            window = np.concatenate(
                [self._buffer, np.zeros(pad_len, dtype=np.float32)]
            )
            if self.incremental:
                result = self._detector.process_stream_events(
                    window, start_frame=self._frame_idx
                )
            else:
                result = self._detector.process_stream(
                    window, start_frame=self._frame_idx
                )
            all_results.extend(result)
            self._frame_idx += 1
            self._frames_processed += 1
            self._buffer = np.array([], dtype=np.float32)

        end_time = self._frame_idx * self.hop_seconds
        if self.incremental:
            flush_result = self._detector.flush_stream_events(end_time)
        else:
            flush_result = self._detector.flush_stream(end_time)
        all_results.extend(flush_result)

        return all_results

    @property
    def frames_processed(self) -> int:
        """Number of windows processed so far (for reporting)."""
        return self._frames_processed
