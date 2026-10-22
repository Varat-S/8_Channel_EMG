"""Traditional time-domain EMG features."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .types import CHANNEL_COUNT


FEATURES_PER_CHANNEL = ("mav", "wl", "zc", "ssc")


def extract_time_domain_features(
    window: ArrayLike,
    thresholds: float | ArrayLike = 0.0,
) -> NDArray[np.float64]:
    """Return MAV, WL, ZC, and SSC for each channel (channel-major order).

    ``thresholds`` are minimum changes in the same unit as the samples. They
    suppress sign changes due only to small fluctuations around zero.
    """

    samples = np.asarray(window, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != CHANNEL_COUNT:
        raise ValueError(f"window must have shape (samples, {CHANNEL_COUNT})")
    if samples.shape[0] < 3:
        raise ValueError("at least three samples are required")

    threshold = np.broadcast_to(
        np.asarray(thresholds, dtype=np.float64), (CHANNEL_COUNT,)
    )
    if np.any(threshold < 0):
        raise ValueError("feature thresholds cannot be negative")

    differences = np.diff(samples, axis=0)
    mav = np.mean(np.abs(samples), axis=0)
    waveform_length = np.sum(np.abs(differences), axis=0)

    zero_crossing_mask = (samples[:-1] * samples[1:] < 0) & (
        np.abs(differences) >= threshold
    )
    zero_crossings = np.sum(zero_crossing_mask, axis=0)

    left_slope = differences[:-1]
    right_slope = differences[1:]
    slope_change_mask = (left_slope * right_slope < 0) & (
        np.maximum(np.abs(left_slope), np.abs(right_slope)) >= threshold
    )
    slope_sign_changes = np.sum(slope_change_mask, axis=0)

    return np.column_stack(
        (mav, waveform_length, zero_crossings, slope_sign_changes)
    ).reshape(-1)


def feature_names() -> list[str]:
    return [
        f"CH{channel}_{feature}"
        for channel in range(CHANNEL_COUNT)
        for feature in FEATURES_PER_CHANNEL
    ]

