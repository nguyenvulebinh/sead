# SEAD - Sound Event Activity Detection

Sound Event Activity Detection (SEAD). Detects speech, music, and other sounds in audio with offline and streaming support.

## Installation

```bash
pip install sead
```

For microphone streaming, also install the optional `sounddevice` dependency:

```bash
pip install sead[dev]
```

For GPU inference (optional):

```bash
pip install sead[onnx-gpu]
```

## Usage

### CLI

```bash
# Offline: process a WAV file
sead --audio audio_16khz.wav

# Or via Python module
python -m sead.cli --audio audio_16khz.wav

# Synthetic streaming: feed file as streaming input (emits start/end events)
sead --stream-file audio_16khz.wav

# Compare streaming vs offline
sead --stream-file audio_16khz.wav --compare

# Live microphone streaming (requires sounddevice)
sead --stream

# Save debug chunks for each segment
sead --audio audio_16khz.wav --debug-dir debug_segments
```

### Python API

```python
from sead import DEFAULT_MODEL_PATH, SEADDetector, SEADIterator, Segment
from pathlib import Path

# Offline
detector = SEADDetector(DEFAULT_MODEL_PATH)
segments = detector.process_file(Path("audio.wav"))
for s in segments:
    print(s)  # [start_time, end_time, label, confidence]

# Streaming (incremental events)
iterator = SEADIterator(detector)
for chunk in audio_chunks:
    for e in iterator(chunk):
        print(e)  # {'start': t, 'label': str} or {'end': t, 'label': str, 'confidence': float}
for e in iterator.flush():
    print(e)
```

## Output

- **Segments**: `[start_time, end_time, label, confidence]` with labels `speech`, `music`, `others`
- **Events** (streaming): `{'start': t, 'label': str}` on onset, `{'end': t, 'label': str, 'confidence': float}` on offset

## Acknowledgement

SEAD uses [YamNet](https://huggingface.co/qualcomm/YamNet) by Qualcomm, an audio event classifier trained on the AudioSet dataset.