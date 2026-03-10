"""Sound Event Detection (SEAD) module."""

from importlib.metadata import version

try:
    __version__ = version("sead")
except Exception:
    __version__ = "0.0.0"

from sead.config import DEFAULT_CLASS_MAP_PATH, DEFAULT_MODEL_PATH, Segment
from sead.detector import SEADDetector
from sead.iterator import SEADIterator

__all__ = [
    "__version__",
    "DEFAULT_CLASS_MAP_PATH",
    "DEFAULT_MODEL_PATH",
    "SEADDetector",
    "SEADIterator",
    "Segment",
]
