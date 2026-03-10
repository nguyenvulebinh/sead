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
from sead.config import DEFAULT_MODEL_PATH
from sead.iterator import SEADIterator


def main() -> None:
    parser = argparse.ArgumentParser(description="SEADIterator streaming demo")
    parser.add_argument("--audio", type=str, required=True, help="Path to WAV file")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    iterator = SEADIterator(DEFAULT_MODEL_PATH, sampling_rate=SAMPLE_RATE)
    window_samples = iterator.window_samples
    hop_samples = iterator.hop_samples
    all_segments = []
    buffer = np.array([], dtype=np.float32)
    chunk_count = 0

    t0 = time.perf_counter()
    with sf.SoundFile(str(audio_path)) as f:
        if f.samplerate != SAMPLE_RATE:
            raise ValueError(
                f"Streaming demo expects {SAMPLE_RATE}Hz, file is {f.samplerate}Hz"
            )
        while True:
            chunk = f.read(frames=hop_samples, dtype="float32")
            if len(chunk) == 0:
                break
            if chunk.ndim > 1:
                chunk = chunk.mean(axis=1)
            buffer = np.concatenate([buffer, chunk.astype(np.float32)])
            while len(buffer) >= window_samples:
                window = buffer[:window_samples].copy()
                buffer = buffer[hop_samples:]
                segments = iterator(window)
                chunk_count += 1
                all_segments.extend(segments)
                if len(segments) > 0:
                    for s in segments:
                        print(s)

    flush_segments = iterator.flush()
    if len(flush_segments) > 0:
        for s in flush_segments:
            print(s)
    all_segments.extend(flush_segments)
    wall_time = time.perf_counter() - t0

    print(f"Processed {audio_path.name} with SEADIterator (streaming simulation)")
    print(f"Incremental events (start/end, VADIterator-style): {len(all_segments)}")
    print()
    print("=== Report ===")
    print(f"  Wall time:       {wall_time:.2f} s")
    if chunk_count > 0:
        print(f"  Chunks/sec:      {chunk_count / wall_time:.1f}")
        print(f"  Latency (ms):    mean {wall_time / chunk_count * 1000:.1f}")

if __name__ == "__main__":
    main()
