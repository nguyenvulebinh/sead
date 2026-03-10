"""SEAD configuration: constants, class mappings, and thresholds."""

from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16000


def _get_default_model_path() -> Path:
    """Load bundled model path via importlib.resources (portable across installs)."""
    package_path = "sead.data"
    model_name = "YamNet_w8a8.onnx.zip"
    try:
        import importlib_resources as impresources
        return Path(impresources.files(package_path).joinpath(model_name))
    except ImportError:
        from importlib import resources as impresources
        with impresources.path(package_path, model_name) as f:
            return Path(f)


# Default model path (bundled in sead/data/)
DEFAULT_MODEL_PATH = _get_default_model_path()

# Default max CPU threads for ONNX and PyTorch (2 for streaming-friendly)
DEFAULT_NUM_THREADS = 2


def _get_default_class_map_path() -> Path:
    """Load bundled class map path via importlib.resources."""
    package_path = "sead.data"
    name = "yamnet_class_map.csv"
    try:
        import importlib_resources as impresources
        return Path(impresources.files(package_path).joinpath(name))
    except ImportError:
        from importlib import resources as impresources
        with impresources.path(package_path, name) as f:
            return Path(f)


DEFAULT_CLASS_MAP_PATH = _get_default_class_map_path()
CHUNK_LENGTH_SECONDS = 0.98
PATCH_HOP_SEC = 0.48
PATCH_LEN_SEC = 0.96

EMA_ALPHA = 0.4

# AudioSet class indices -> target coarse classes
SPEECH_INDICES = frozenset(
    list(range(0, 6))  # Speech(0) .. Speech synthesizer(5)
    + [65]  # Hubbub, speech noise, speech babble
)
MUSIC_INDICES = frozenset(
    list(range(24, 33))  # Singing(24) .. Humming(32)
    + [35]  # Whistling
    + list(range(132, 277))  # Music(132) .. Scary music(276)
    + list(range(456, 459))  # Electronic tuner(456) .. Chorus effect(458)
)

TARGET_LABELS = ("speech", "music", "others")

# Temporal decoding
ONSET_THRESHOLD = 0.5
OFFSET_THRESHOLD = 0.35
MIN_DURATION_SEC = 0.5
MAX_GAP_SEC = 0.3


@dataclass
class Segment:
    """A detected sound event segment."""

    start_time: float
    end_time: float
    label: str
    confidence: float

    def __str__(self) -> str:
        return f"[{self.start_time:.2f}, {self.end_time:.2f}, {self.label}, {self.confidence:.3f}]"
