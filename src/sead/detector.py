"""Main SEAD detector: offline and streaming sound event detection."""

import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

from sead.audio_utils import (
    PATCH_HOP_SEC as AUDIO_PATCH_HOP,
    chunk_and_resample_audio,
    load_audio_wav,
    waveform_to_yamnet_patches,
)
from sead.class_mapping import aggregate_to_target_classes
from sead.config import (
    EMA_ALPHA,
    MAX_GAP_SEC,
    MIN_DURATION_SEC,
    PATCH_HOP_SEC,
    Segment,
)
from sead.model_utils import extract_onnx_from_zip, run_yamnet_onnx
from sead.segment_builder import build_segments
from sead.smoothing import EMASmoother
from sead.temporal_decoder import TemporalDecoder


class SEADDetector:
    """Sound Event Detection: YAMNet + class aggregation + EMA + hysteresis + segments."""

    def __init__(
        self,
        model_zip_path: Path,
        *,
        ema_alpha: float = EMA_ALPHA,
        onset_threshold: float = 0.5,
        offset_threshold: float = 0.35,
        min_duration_sec: float = MIN_DURATION_SEC,
        max_gap_sec: float = MAX_GAP_SEC,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.min_duration_sec = min_duration_sec
        self.max_gap_sec = max_gap_sec

        self._tmpdir = tempfile.mkdtemp(prefix="sead_onnx_")
        onnx_path = extract_onnx_from_zip(
            Path(model_zip_path).expanduser().resolve(),
            Path(self._tmpdir),
        )
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        self._smoother = EMASmoother(ema_alpha, num_classes=3)
        self._decoder = TemporalDecoder(
            onset_threshold=onset_threshold,
            offset_threshold=offset_threshold,
            frame_hop_sec=PATCH_HOP_SEC,
        )

    def __del__(self) -> None:
        if hasattr(self, "_tmpdir"):
            import shutil

            try:
                shutil.rmtree(self._tmpdir, ignore_errors=True)
            except Exception:
                pass

    def process_file(self, audio_path: Path) -> list[Segment]:
        """Process audio file offline. Returns list of segments."""
        audio, sr = load_audio_wav(Path(audio_path).expanduser().resolve())
        chunks = chunk_and_resample_audio(
            audio, sr, hop_seconds=AUDIO_PATCH_HOP
        )

        all_logits: list[np.ndarray] = []
        for chunk in chunks:
            patches = waveform_to_yamnet_patches(chunk)
            if patches.size == 0:
                continue
            logits = run_yamnet_onnx(self._session, patches)
            all_logits.append(logits)

        if not all_logits:
            return []

        logits = np.concatenate([p.reshape(-1, p.shape[-1]) for p in all_logits], axis=0)
        probs = aggregate_to_target_classes(logits)

        self._smoother.reset()
        smoothed = self._smoother.update(probs)

        self._decoder.reset()
        raw = self._decoder.decode(smoothed)
        end_time = (smoothed.shape[0] - 1) * PATCH_HOP_SEC + PATCH_HOP_SEC
        raw.extend(self._decoder.flush(end_time))

        return build_segments(
            raw,
            min_duration_sec=self.min_duration_sec,
            max_gap_sec=self.max_gap_sec,
        )

    def process_stream(
        self,
        waveform: np.ndarray,
        start_frame: int,
    ) -> list[Segment]:
        """
        Process a single streaming chunk. Returns completed segments only.

        Args:
            waveform: mono float32 audio chunk (e.g. 0.48s or 0.96s)
            start_frame: frame index offset for this chunk

        Returns:
            Segments that have ended (offset detected) in this chunk
        """
        patches = waveform_to_yamnet_patches(waveform)
        if patches.size == 0:
            return []

        logits = run_yamnet_onnx(self._session, patches)
        probs = aggregate_to_target_classes(logits)
        smoothed = self._smoother.update(probs)

        raw = self._decoder.decode(smoothed, start_frame=start_frame)

        return build_segments(
            raw,
            min_duration_sec=self.min_duration_sec,
            max_gap_sec=self.max_gap_sec,
        )

    def reset_stream(self) -> None:
        """Reset smoother and decoder for new stream."""
        self._smoother.reset()
        self._decoder.reset()

    def flush_stream(self, end_time: float) -> list[Segment]:
        """Flush any active segment at end of stream. Call after last process_stream."""
        raw = self._decoder.flush(end_time)
        return build_segments(
            raw,
            min_duration_sec=self.min_duration_sec,
            max_gap_sec=self.max_gap_sec,
        )

    def process_stream_events(
        self,
        waveform: np.ndarray,
        start_frame: int,
    ) -> list[dict]:
        """
        Process a single streaming chunk. Returns incremental start/end events
        (VADIterator-style): {'start': t, 'label': str} or
        {'end': t, 'label': str, 'confidence': float}.
        """
        patches = waveform_to_yamnet_patches(waveform)
        if patches.size == 0:
            return []

        logits = run_yamnet_onnx(self._session, patches)
        probs = aggregate_to_target_classes(logits)
        smoothed = self._smoother.update(probs)

        return self._decoder.decode_events(smoothed, start_frame=start_frame)

    def flush_stream_events(self, end_time: float) -> list[dict]:
        """Flush any active segment as end event. Call after last process_stream_events."""
        return self._decoder.flush_events(end_time)
