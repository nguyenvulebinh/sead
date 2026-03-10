"""Aggregate AudioSet 521 classes to coarse target classes."""

import numpy as np
from scipy.special import expit as sigmoid

from sead.config import MUSIC_INDICES, SPEECH_INDICES, TARGET_LABELS


def _build_aggregation_matrix() -> np.ndarray:
    """Build [521, 3] matrix: column k sums probs for target class k."""
    n_audioset = 521
    matrix = np.zeros((n_audioset, 3), dtype=np.float32)
    for i in range(n_audioset):
        if i in SPEECH_INDICES:
            matrix[i, 0] = 1.0
        elif i in MUSIC_INDICES:
            matrix[i, 1] = 1.0
        else:
            matrix[i, 2] = 1.0
    return matrix


_AGG_MATRIX = _build_aggregation_matrix()


def aggregate_to_target_classes(logits: np.ndarray) -> np.ndarray:
    """
    Aggregate 521-dim AudioSet logits to 3 target classes (speech, music, others).

    target_prob[k] = sum(prob[i] for i in class_ids_k), then normalize so rows sum to 1.

    Args:
        logits: [N, 521] raw YAMNet logits

    Returns:
        [N, 3] target probabilities (speech, music, others), each row sums to 1
    """
    probs = sigmoid(logits.astype(np.float64))
    out = probs @ _AGG_MATRIX
    row_sum = out.sum(axis=1, keepdims=True)
    out = np.where(row_sum > 0, out / row_sum, out)
    return out.astype(np.float32)


def get_target_labels() -> tuple[str, ...]:
    """Return target class labels in order."""
    return TARGET_LABELS
