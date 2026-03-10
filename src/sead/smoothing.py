"""Exponential Moving Average for probability smoothing."""

import numpy as np


class EMASmoother:
    """Causal EMA: P_smooth[t] = alpha * P[t] + (1-alpha) * P_smooth[t-1]."""

    def __init__(self, alpha: float, num_classes: int) -> None:
        self.alpha = alpha
        self.num_classes = num_classes
        self._state: np.ndarray | None = None

    def update(self, probs: np.ndarray) -> np.ndarray:
        """
        Update state and return smoothed probabilities.

        Args:
            probs: [N, num_classes] or [num_classes] target probabilities

        Returns:
            Smoothed probs, same shape as probs
        """
        single = probs.ndim == 1
        if single:
            probs = probs.reshape(1, -1)

        if probs.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, got {probs.shape[1]}"
            )

        out = np.empty_like(probs, dtype=np.float32)
        for i in range(probs.shape[0]):
            p = probs[i]
            if self._state is None:
                self._state = p.copy()
            else:
                self._state = self.alpha * p + (1.0 - self.alpha) * self._state
            out[i] = self._state.copy()

        if single:
            return out[0]
        return out

    def reset(self) -> None:
        """Reset internal state (e.g. for new audio stream)."""
        self._state = None
