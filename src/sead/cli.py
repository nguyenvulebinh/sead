"""CLI for SEAD sound event detection."""

import argparse
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from sead.audio_utils import SAMPLE_RATE, load_audio_wav
from sead.config import DEFAULT_MODEL_PATH, DEFAULT_NUM_THREADS, PATCH_HOP_SEC
from sead.detector import SEADDetector
from sead.iterator import SEADIterator


def _run_stream(detector: SEADDetector) -> None:
    """Stream from microphone, print completed segments."""
    try:
        import sounddevice as sd
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Streaming requires 'sounddevice'. Install it and try again."
        ) from e

    window_sec = 0.98
    hop_sec = 0.48
    block_ms = 50
    window_samples = int(round(window_sec * SAMPLE_RATE))
    hop_samples = int(round(hop_sec * SAMPLE_RATE))
    blocksize = max(1, int(round((block_ms / 1000.0) * SAMPLE_RATE)))

    ring = np.zeros((window_samples,), dtype="float32")
    write_pos = 0
    total_written = 0
    samples_since_last_infer = 0
    lock = threading.Lock()
    stop_event = threading.Event()

    def _ring_write(x):
        nonlocal ring, write_pos, total_written, samples_since_last_infer
        n = int(x.shape[0])
        if n <= 0:
            return
        if n >= ring.shape[0]:
            ring[:] = x[-ring.shape[0] :]
            write_pos = 0
        else:
            end = write_pos + n
            if end <= ring.shape[0]:
                ring[write_pos:end] = x
            else:
                first = ring.shape[0] - write_pos
                ring[write_pos:] = x[:first]
                ring[: end - ring.shape[0]] = x[first:]
            write_pos = end % ring.shape[0]
        total_written += n
        samples_since_last_infer += n

    def _ring_read():
        if write_pos == 0:
            return ring.copy()
        return np.concatenate((ring[write_pos:], ring[:write_pos])).copy()

    def callback(indata, frames, _time, status):
        if status:
            print(f"[audio] {status}", flush=True)
        if stop_event.is_set():
            raise sd.CallbackStop()
        x = np.asarray(indata, dtype="float32").reshape(-1)
        with lock:
            _ring_write(x)

    detector.reset_stream()
    frame_idx = 0

    print(
        f"Streaming @ {SAMPLE_RATE}Hz | window={window_sec}s hop={hop_sec}s | Ctrl+C to stop"
    )
    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    ):
        try:
            while True:
                do_infer = False
                with lock:
                    if total_written >= window_samples and samples_since_last_infer >= hop_samples:
                        samples_since_last_infer = 0
                        waveform = _ring_read()
                        do_infer = True

                if do_infer:
                    segments = detector.process_stream(waveform, start_frame=frame_idx)
                    for s in segments:
                        print(s)
                    frame_idx += 1

                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()


def _run_synthetic_stream(
    detector: SEADDetector,
    audio_path: Path,
    *,
    compare: bool = False,
    on_event: Callable | None = None,
) -> list:
    """
    Simulate streaming by feeding a WAV file in hop-sized chunks.
    SEADIterator buffers internally until a full window is available.

    If on_event is provided, call it for each segment/event as it is detected
    (enables real-time output during streaming).
    """
    audio, sr = load_audio_wav(audio_path)
    if sr != SAMPLE_RATE:
        from sead.audio_utils import chunk_and_resample_audio

        chunks = chunk_and_resample_audio(audio, sr)
        audio = np.concatenate(chunks, axis=0)

    iterator = SEADIterator(
        detector, sampling_rate=SAMPLE_RATE, incremental=not compare
    )
    all_segments: list = []
    hop_samples = iterator.hop_samples
    start = 0

    t0 = time.perf_counter()
    while start < audio.shape[0]:
        end = min(start + hop_samples, audio.shape[0])
        waveform = audio[start:end].astype(np.float32)
        segments = iterator(waveform)
        all_segments.extend(segments)
        if on_event and segments:
            for s in segments:
                on_event(s)
        start += hop_samples

    flushed = iterator.flush()
    all_segments.extend(flushed)
    if on_event and flushed:
        for s in flushed:
            on_event(s)
    wall_time = time.perf_counter() - t0
    frames_processed = iterator.frames_processed

    print()
    print("=== Report ===")
    print(f"  Wall time:       {wall_time:.2f} s")
    if frames_processed > 0:
        print(f"  Chunks/sec:      {frames_processed / wall_time:.1f}")
        print(f"  Latency (ms):    mean {wall_time / frames_processed * 1000:.1f}")

    if compare:
        offline = detector.process_file(audio_path)
        return all_segments, offline
    return all_segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sound Event Detection (SEAD) - offline and streaming"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audio", type=str, help="Path to WAV file (offline)")
    mode.add_argument("--stream", action="store_true", help="Stream from microphone")
    mode.add_argument(
        "--stream-file",
        type=str,
        metavar="WAV",
        help="Synthetic streaming: feed WAV file as streaming input (compare with offline)",
    )
    parser.add_argument(
        "--model-zip",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to YamNet ONNX zip",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra debug info",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Save audio chunks for each segment to DIR for debugging (offline only)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="With --stream-file: print streaming vs offline comparison",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        metavar="N",
        help=f"Max CPU threads for inference (default: {DEFAULT_NUM_THREADS})",
    )
    args = parser.parse_args()

    model_path = Path(args.model_zip).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    num_threads = args.num_threads if args.num_threads is not None else DEFAULT_NUM_THREADS
    detector = SEADDetector(model_path, num_threads=num_threads)

    if args.stream:
        _run_stream(detector)
        return

    if args.stream_file:
        audio_path = Path(args.stream_file).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        if args.compare:
            stream_segs, offline_segs = _run_synthetic_stream(
                detector, audio_path, compare=True
            )
            print("=== STREAMING (synthetic) ===")
            print(f"Detected {len(stream_segs)} segments:")
            for s in stream_segs:
                print(s)
            print()
            print("=== OFFLINE ===")
            print(f"Detected {len(offline_segs)} segments:")
            for s in offline_segs:
                print(s)
            print()
            print("=== COMPARISON ===")
            print(f"Streaming: {len(stream_segs)} segments")
            print(f"Offline:   {len(offline_segs)} segments")
        else:
            stream_segs = _run_synthetic_stream(
                detector, audio_path, on_event=print
            )
            print(f"Synthetic streaming: {len(stream_segs)} events")
        return

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    with sf.SoundFile(str(audio_path)) as f:
        audio_duration_sec = len(f) / f.samplerate
    num_chunks = max(1, int((audio_duration_sec - 0.98) / PATCH_HOP_SEC) + 1)

    t0 = time.perf_counter()
    segments = detector.process_file(audio_path)
    wall_time = time.perf_counter() - t0

    print(f"Detected {len(segments)} segments:")
    for s in segments:
        print(s)

    print()
    print("=== Report ===")
    print(f"  Wall time:       {wall_time:.2f} s")
    print(f"  Chunks/sec:      {num_chunks / wall_time:.1f}")
    print(f"  Latency (ms):    mean {wall_time / num_chunks * 1000:.1f}")

    if args.debug_dir:
        debug_path = Path(args.debug_dir).expanduser().resolve()
        debug_path.mkdir(parents=True, exist_ok=True)
        audio, sr = load_audio_wav(audio_path)
        for i, s in enumerate(segments):
            start_sample = int(s.start_time * sr)
            end_sample = int(s.end_time * sr)
            chunk = audio[start_sample:end_sample]
            safe_label = s.label.replace(" ", "_")
            filename = f"{i:03d}_{s.start_time:.1f}s_{s.end_time:.1f}s_{safe_label}.wav"
            out_path = debug_path / filename
            sf.write(str(out_path), chunk, sr)
        print(f"Saved {len(segments)} chunks to {debug_path}")


if __name__ == "__main__":
    main()
