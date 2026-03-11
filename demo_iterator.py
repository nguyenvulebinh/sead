"""
Demo: process audio file using SEADIterator (VADIterator-like streaming interface).

Simulates streaming by reading the file in chunks (hop-sized) and feeding
them to the iterator as they "arrive" - no full-file load or chunk_and_resample.

Usage:
    python demo_iterator.py --audio path/to/audio.wav
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from sead.audio_utils import SAMPLE_RATE
from sead.config import DEFAULT_MODEL_PATH, DEFAULT_NUM_THREADS
from sead.iterator import SEADIterator


def main() -> None:
    parser = argparse.ArgumentParser(description="SEADIterator streaming demo")
    parser.add_argument("--audio", type=str, required=True, help="Path to WAV file")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        metavar="N",
        help=f"Max CPU threads for inference (default: {DEFAULT_NUM_THREADS})",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    num_threads = args.num_threads if args.num_threads is not None else DEFAULT_NUM_THREADS
    iterator = SEADIterator(
        DEFAULT_MODEL_PATH, sampling_rate=SAMPLE_RATE, num_threads=num_threads
    )
    all_events: list = []

    t0 = time.perf_counter()
    with sf.SoundFile(str(audio_path)) as f:
        if f.samplerate != SAMPLE_RATE:
            raise ValueError(
                f"Streaming demo expects {SAMPLE_RATE}Hz, file is {f.samplerate}Hz"
            )
        while True:
            chunk = f.read(frames=iterator.hop_samples, dtype="float32")
            if len(chunk) == 0:
                break
            if chunk.ndim > 1:
                chunk = chunk.mean(axis=1)
            segments = iterator(chunk.astype(np.float32))
            all_events.extend(segments)
            if len(segments) > 0:
                for s in segments:
                    print(s)

    flush_segments = iterator.flush()
    all_events.extend(flush_segments)
    if len(flush_segments) > 0:
        for s in flush_segments:
            print(s)

    wall_time = time.perf_counter() - t0
    frames_processed = iterator.frames_processed

    print(f"Processed {audio_path.name} with SEADIterator (streaming simulation)")
    print(f"Incremental events (start/end, VADIterator-style): {len(all_events)}")
    print()
    print("=== Report ===")
    print(f"  Wall time:       {wall_time:.2f} s")
    if frames_processed > 0:
        print(f"  Frames processed: {frames_processed}")
        print(f"  Frames/sec:      {frames_processed / wall_time:.1f}")
        print(f"  Latency (ms):    mean {wall_time / frames_processed * 1000:.1f}")

if __name__ == "__main__":
    main()
