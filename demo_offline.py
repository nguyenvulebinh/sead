"""
Demo: offline processing of an audio file.

Usage:
    python demo_offline.py --audio path/to/audio.wav
"""

import argparse
import time
from pathlib import Path

import soundfile as sf

from sead.config import DEFAULT_MODEL_PATH, PATCH_HOP_SEC
from sead.detector import SEADDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="SEAD offline demo")
    parser.add_argument("--audio", type=str, required=True, help="Path to WAV file")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    with sf.SoundFile(str(audio_path)) as f:
        audio_duration_sec = len(f) / f.samplerate

    num_chunks = max(1, int((audio_duration_sec - 0.98) / PATCH_HOP_SEC) + 1)

    detector = SEADDetector(DEFAULT_MODEL_PATH)
    t0 = time.perf_counter()
    segments = detector.process_file(audio_path)
    wall_time = time.perf_counter() - t0

    print(f"Input: {audio_path}")
    print(f"Detected {len(segments)} segments:")
    for s in segments:
        print(s)

    print()
    print("=== Report ===")
    print(f"  Wall time:       {wall_time:.2f} s")
    print(f"  Chunks/sec:      {num_chunks / wall_time:.1f}")
    print(f"  Latency (ms):    mean {wall_time / num_chunks * 1000:.1f}")


if __name__ == "__main__":
    main()
