"""Audio loading and preprocessing (self-contained, no dependency on demo.py)."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
CHUNK_LENGTH_SECONDS = 0.98


def load_audio_wav(path: Path) -> tuple[np.ndarray, int]:
    """
    Returns mono float32 waveform in range [-1, 1], shape [num_samples].
    """
    x, sr = sf.read(str(path), dtype="int16", always_2d=True)
    x = (x.astype(np.float32) / 2**15).T
    if x.shape[0] > 1:
        x = np.mean(x, axis=0, keepdims=False)
    else:
        x = x[0]
    return x.astype(np.float32), int(sr)


# YAMNet patch: 0.96s window, 0.48s hop
PATCH_HOP_SEC = 0.48
PATCH_LEN_SEC = 0.96


def chunk_and_resample_audio(
    audio: np.ndarray,
    audio_sample_rate: int,
    model_sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = CHUNK_LENGTH_SECONDS,
    hop_seconds: float | None = None,
) -> list[np.ndarray]:
    """
    Split audio into chunks, resampling if needed.

    Uses overlapping windows (hop_seconds < chunk_seconds) to ensure full
    temporal coverage for SEAD. Default hop matches YAMNet patch hop (0.48s).
    """
    if audio_sample_rate != model_sample_rate:
        try:
            import resampy  # type: ignore

            audio = resampy.resample(
                audio, audio_sample_rate, model_sample_rate
            ).astype(np.float32)
        except ModuleNotFoundError:
            import math

            g = math.gcd(int(audio_sample_rate), int(model_sample_rate))
            up = int(model_sample_rate // g)
            down = int(audio_sample_rate // g)
            audio = resample_poly(audio, up=up, down=down).astype(np.float32)
        audio_sample_rate = model_sample_rate

    chunk_len = int(round(audio_sample_rate * chunk_seconds))
    hop = int(round(audio_sample_rate * (hop_seconds or chunk_seconds)))
    if chunk_len <= 0:
        return [audio]
    if audio.shape[0] <= chunk_len:
        return [audio]
    hop = max(1, hop)

    chunks: list[np.ndarray] = []
    start = 0
    while start < audio.shape[0]:
        end = min(start + chunk_len, audio.shape[0])
        segment = audio[start:end]
        if segment.size < chunk_len:
            segment = np.pad(
                segment, (0, chunk_len - segment.size), mode="constant", constant_values=0
            )
        chunks.append(segment.astype(np.float32))
        if end >= audio.shape[0]:
            break
        start += hop
    return chunks


def waveform_to_yamnet_patches(waveform: np.ndarray) -> np.ndarray:
    """
    Converts waveform to YAMNet log-mel patches.

    Returns numpy float32 array shaped like [N, 1, 96, 64].
    """
    import torch
    import torch.nn.functional as F
    import torchaudio

    win_length = int(round(0.025 * SAMPLE_RATE))
    hop_length = int(round(0.010 * SAMPLE_RATE))
    n_fft = 512
    n_mels = 64

    wf = torch.from_numpy(waveform).to(torch.float32).reshape(1, -1)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=125.0,
        f_max=7500.0,
        n_mels=n_mels,
        power=1.0,
        center=False,
        mel_scale="htk",
        norm=None,
    )(wf)
    log_mel = torch.log(mel + 1e-6).squeeze(0).transpose(0, 1)

    frames_per_patch = 96
    patch_hop_frames = 48

    if log_mel.shape[0] < frames_per_patch:
        log_mel = F.pad(
            log_mel, (0, 0, 0, frames_per_patch - log_mel.shape[0])
        )

    patches = []
    for start in range(0, log_mel.shape[0] - frames_per_patch + 1, patch_hop_frames):
        patch = log_mel[start : start + frames_per_patch, :]
        patches.append(patch.unsqueeze(0).unsqueeze(0))

    if not patches:
        raise RuntimeError("Failed to generate patches from audio.")
    return torch.cat(patches, dim=0).cpu().numpy().astype(np.float32)
