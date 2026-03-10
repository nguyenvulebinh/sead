"""YAMNet ONNX model loading and inference (self-contained, no dependency on demo.py)."""

import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort


def extract_onnx_from_zip(model_zip_path: Path, extract_dir: Path) -> Path:
    """Extract ONNX file from zip bundle."""
    if not model_zip_path.exists():
        raise FileNotFoundError(f"Model zip not found: {model_zip_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(model_zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        shutil.unpack_archive(str(model_zip_path), str(extract_dir))

    onnx_files = sorted(extract_dir.rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(
            f"No .onnx found after extracting {model_zip_path} into {extract_dir}"
        )
    return onnx_files[0]


def run_yamnet_onnx(session: ort.InferenceSession, patches: np.ndarray) -> np.ndarray:
    """
    Run YAMNet ONNX inference.

    Args:
        session: ONNX inference session
        patches: [N, 1, 96, 64] log-mel patches

    Returns:
        [N, 521] raw logits
    """
    inp = session.get_inputs()[0]
    if patches.ndim != 4 or patches.shape[1:] != (1, 96, 64):
        raise ValueError(f"Expected patches shaped [N,1,96,64], got {patches.shape}")

    outputs: list[np.ndarray] = []
    for i in range(patches.shape[0]):
        feed = {inp.name: patches[i : i + 1]}
        out = session.run(None, feed)
        if not out:
            raise RuntimeError("ONNXRuntime returned no outputs.")
        outputs.append(np.asarray(out[0]).reshape(1, -1))
    return np.concatenate(outputs, axis=0)
