"""
Demo: offline processing of an audio file.

Usage:
    python demo_offline.py --audio path/to/audio.wav
"""

import argparse
from pathlib import Path

from sead.config import DEFAULT_MODEL_PATH
from sead.detector import SEADDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="SEAD offline demo")
    parser.add_argument("--audio", type=str, required=True, help="Path to WAV file")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    detector = SEADDetector(DEFAULT_MODEL_PATH)
    segments = detector.process_file(audio_path)

    print(f"Input: {audio_path}")
    print(f"Detected {len(segments)} segments:")
    for s in segments:
        print(s)


if __name__ == "__main__":
    main()
