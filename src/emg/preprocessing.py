"""Causal preprocessing shared by offline training and live inference."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, sosfilt, sosfilt_zi

from .types import CHANNEL_COUNT


def apply_spatial_reference(
    values: ArrayLike, mode: str = "none"
) -> NDArray[np.float64]:
    """Apply an optional across-channel reference to each simultaneous sample."""

    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim == 1 and samples.shape == (CHANNEL_COUNT,):
        if mode == "none":
            return samples
        if mode == "common_average":
            return samples - np.mean(samples)
        raise ValueError(f"unknown spatial reference mode: {mode}")
    if samples.ndim != 2 or samples.shape[1] != CHANNEL_COUNT:
        raise ValueError(f"values must have shape (samples, {CHANNEL_COUNT})")
    if mode == "none":
        return samples
    if mode == "common_average":
        return samples - np.mean(samples, axis=1, keepdims=True)
    raise ValueError(f"unknown spatial reference mode: {mode}")


def design_bandpass(
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> NDArray[np.float64]:
    nyquist = sample_rate_hz / 2.0
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f"band-pass must satisfy 0 < low < high < {nyquist:g} Hz"
        )
    return butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )


def filter_recording(
    values: ArrayLike,
    *,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> NDArray[np.float64]:
    """Causally filter a continuous recording with steady-state initialization."""

    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != CHANNEL_COUNT:
        raise ValueError(f"values must have shape (samples, {CHANNEL_COUNT})")
    sos = design_bandpass(sample_rate_hz, low_hz, high_hz, order)
    zi = sosfilt_zi(sos)[:, :, None] * samples[0][None, None, :]
    filtered, _ = sosfilt(sos, samples, axis=0, zi=zi)
    return filtered


class CausalBandpass:
    """Sample-by-sample form of the same filter used by ``filter_recording``."""

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        low_hz: float,
        high_hz: float,
        order: int = 4,
    ) -> None:
        self._sos = design_bandpass(sample_rate_hz, low_hz, high_hz, order)
        self._zi: NDArray[np.float64] | None = None

    def update(self, sample: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(sample, dtype=np.float64)
        if values.shape != (CHANNEL_COUNT,):
            raise ValueError(f"sample must have shape ({CHANNEL_COUNT},)")
        if self._zi is None:
            self._zi = sosfilt_zi(self._sos)[:, :, None] * values[None, None, :]
        output, self._zi = sosfilt(
            self._sos, values[None, :], axis=0, zi=self._zi
        )
        return output[0]

    def reset(self) -> None:
        self._zi = None
