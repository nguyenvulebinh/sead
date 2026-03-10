"""YAMNet ONNX model loading and inference (self-contained, no dependency on demo.py)."""

import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort


def get_quantization_params(onnx_path: Path) -> dict[str, float]:
    """
    Extract quantization params from ONNX model (audio_scale, audio_zero_point,
    class_scores_scale, class_scores_zero_point). Returns empty dict for float models.
    """
    result: dict[str, float] = {}
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        return result

    model = onnx.load(str(onnx_path))
    for init in model.graph.initializer:
        if init.name == "audio_scale":
            result["input_scale"] = float(numpy_helper.to_array(init))
        elif init.name == "audio_zero_point":
            result["input_zp"] = float(numpy_helper.to_array(init))
        elif init.name == "class_scores_scale":
            result["output_scale"] = float(numpy_helper.to_array(init))
        elif init.name == "class_scores_zero_point":
            result["output_zp"] = float(numpy_helper.to_array(init))
    return result


def _convert_patches(
    patches: np.ndarray,
    input_dtype: str,
    input_scale: float,
    input_zp: float,
) -> np.ndarray:
    """Convert float32 patches to quantized input (uint8/uint16)."""
    quantized = np.round(patches / input_scale + input_zp).astype(np.float64)
    if "uint8" in input_dtype or "int8" in input_dtype:
        return np.clip(quantized, 0, 255).astype(np.uint8)
    if "uint16" in input_dtype or "int16" in input_dtype:
        return np.clip(quantized, 0, 65535).astype(np.uint16)
    return patches.astype(np.float32)


def _dequantize_logits(
    logits: np.ndarray, output_scale: float, output_zp: float
) -> np.ndarray:
    """Dequantize model output: float = scale * (quantized - zero_point)."""
    return (output_scale * (logits.astype(np.float64) - output_zp)).astype(np.float32)


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


def run_yamnet_onnx(
    session: ort.InferenceSession,
    patches: np.ndarray,
    *,
    quant_params: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Run YAMNet ONNX inference.

    Args:
        session: ONNX inference session
        patches: [N, 1, 96, 64] log-mel patches (float32)
        quant_params: Optional dict with input_scale, input_zp, output_scale, output_zp
            for quantized models (w8a8, w8a16). Omit for float models.

    Returns:
        [N, 521] raw logits (float32)
    """
    inp = session.get_inputs()[0]
    if patches.ndim != 4 or patches.shape[1:] != (1, 96, 64):
        raise ValueError(f"Expected patches shaped [N,1,96,64], got {patches.shape}")

    dtype_str = inp.type if isinstance(inp.type, str) else str(inp.type)
    feed_patches = patches

    if quant_params and "input_scale" in quant_params and "input_zp" in quant_params:
        feed_patches = _convert_patches(
            patches,
            dtype_str,
            quant_params["input_scale"],
            quant_params["input_zp"],
        )

    feed = {inp.name: feed_patches}
    out = session.run(None, feed)
    if not out:
        raise RuntimeError("ONNXRuntime returned no outputs.")

    logits = np.asarray(out[0]).reshape(-1, out[0].shape[-1])

    if quant_params and "output_scale" in quant_params and "output_zp" in quant_params:
        logits = _dequantize_logits(
            logits,
            quant_params["output_scale"],
            quant_params["output_zp"],
        )

    return logits.astype(np.float32)
