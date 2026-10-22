"""Deterministic acquisition-level stressors on (samples, 8) raw signals."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

import numpy as np

from ..types import CHANNEL_COUNT


LEGACY_CORRUPTION_CONFIG = {
    "broadband_snr_db": [20.0, 10.0, 5.0, 0.0],
    "powerline_amplitude_rms": [0.10, 0.25, 0.50, 1.00],
    "powerline_frequency_hz": 50.0,
    "single_channel_noise_snr_db": 0.0,
    "default_repeats": 5,
}

CORRUPTION_CONFIG = {
    "broadband_snr_db": [30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0],
    "powerline_amplitude_rms": [0.01, 0.025, 0.05, 0.10, 0.25, 0.50],
    "powerline_frequency_hz": 50.0,
    "single_channel_noise_snr_db": [20.0, 10.0, 5.0, 0.0],
    "power_reference": "ac",
    "default_repeats": 5,
}


def resolved_corruption_config(version="v1.1", channel_noise_snr=None) -> dict:
    if version not in ("v1", "v1.1"):
        raise ValueError("corruption version must be v1 or v1.1")
    config = dict(LEGACY_CORRUPTION_CONFIG if version == "v1" else CORRUPTION_CONFIG)
    config["power_reference"] = "raw" if version == "v1" else "ac"
    if channel_noise_snr is not None:
        if version == "v1":
            raise ValueError("custom channel-noise sweeps require v1.1")
        if not channel_noise_snr or len(set(channel_noise_snr)) != len(channel_noise_snr) or not np.all(np.isfinite(channel_noise_snr)):
            raise ValueError("channel-noise SNR levels must be finite, nonempty and unique")
        config["single_channel_noise_snr_db"] = list(channel_noise_snr)
    return config


class Corruption(Protocol):
    name: str
    stochastic: bool

    def apply(self, samples: np.ndarray, sampling_rate: float,
              rng: np.random.Generator) -> np.ndarray: ...


def _samples(samples, sampling_rate) -> np.ndarray:
    values = np.array(samples, dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[1] != CHANNEL_COUNT or not len(values):
        raise ValueError(f"samples must have nonempty shape (n_samples, {CHANNEL_COUNT})")
    if not np.all(np.isfinite(values)) or not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("samples and sampling rate must be finite; sampling rate must be positive")
    return values


def signal_power(samples: np.ndarray) -> np.ndarray:
    """Total raw mean-square power per channel, including the DC component."""
    with np.errstate(over="raise", invalid="raise"):
        return np.mean(np.square(samples), axis=0)


def ac_signal_power(samples: np.ndarray) -> np.ndarray:
    """Per-channel mean-removed power, without changing the decoder input.

    A reference shift before calculating the mean is algebraically x - mean(x)
    and avoids rounding the mean of a large constant offset unnecessarily.
    """
    centered = np.asarray(samples, dtype=np.float64) - samples[0]
    centered -= np.mean(centered, axis=0)
    return signal_power(centered)


def reference_power(samples: np.ndarray, reference: str) -> np.ndarray:
    if reference == "ac":
        return ac_signal_power(samples)
    if reference == "raw":
        return signal_power(samples)
    raise ValueError("power_reference must be ac or raw")


def _channels(channels: tuple[int, ...]) -> np.ndarray:
    if (not channels or len(channels) != len(set(channels))
            or any(not isinstance(ch, (int, np.integer)) or not 0 <= ch < CHANNEL_COUNT for ch in channels)):
        raise ValueError("channels must be distinct zero-based indices from 0 through 7")
    return np.array(channels, dtype=int)


@dataclass(frozen=True)
class NoCorruption:
    name: str = "clean"
    stochastic: bool = False

    def apply(self, samples, sampling_rate, rng):
        return _samples(samples, sampling_rate)


@dataclass(frozen=True)
class GaussianNoise:
    snr_db: float
    channels: tuple[int, ...] | None = None
    name: str = "broadband_noise"
    stochastic: bool = True
    power_reference: str = "ac"

    def apply(self, samples, sampling_rate, rng):
        values = _samples(samples, sampling_rate)
        if not np.isfinite(self.snr_db):
            raise ValueError("SNR must be finite")
        channels = _channels(self.channels) if self.channels is not None else np.arange(CHANNEL_COUNT)
        power = reference_power(values[:, channels], self.power_reference)
        # Zero/underflow power means zero added noise; there is no absolute noise floor.
        scale = np.sqrt(power) * np.power(10.0, -self.snr_db / 20.0)
        if not np.all(np.isfinite(scale)):
            raise ValueError("SNR implies non-finite noise amplitude")
        noise = rng.normal(size=(len(values), len(channels))) * scale
        values[:, channels] += noise
        return values


@dataclass(frozen=True)
class PowerlineInterference:
    amplitude_rms: float
    frequency_hz: float = 50.0
    name: str = "powerline_50hz"
    stochastic: bool = True
    power_reference: str = "ac"

    def apply(self, samples, sampling_rate, rng):
        values = _samples(samples, sampling_rate)
        if not np.isfinite(self.amplitude_rms) or self.amplitude_rms < 0:
            raise ValueError("powerline amplitude multiplier must be non-negative and finite")
        if not np.isfinite(self.frequency_hz) or not 0 < self.frequency_hz < sampling_rate / 2:
            raise ValueError("powerline frequency must be below Nyquist and above zero")
        # A is sine PEAK amplitude; injected RMS is approximately A/sqrt(2).
        amplitude = self.amplitude_rms * np.sqrt(reference_power(values, self.power_reference))
        phase = rng.uniform(0, 2 * np.pi, size=CHANNEL_COUNT)
        time = np.arange(len(values), dtype=np.float64) / sampling_rate
        values += amplitude * np.sin(2 * np.pi * self.frequency_hz * time[:, None] + phase)
        return values


@dataclass(frozen=True)
class ChannelDropout:
    channels: tuple[int, ...]
    name: str = "channel_dropout"
    stochastic: bool = False

    def apply(self, samples, sampling_rate, rng):
        values = _samples(samples, sampling_rate)
        values[:, _channels(self.channels)] = 0.0
        return values


@dataclass(frozen=True)
class Condition:
    condition_id: str
    corruption: Corruption
    severity: float | None = None
    channel: str | None = None


def sweep_conditions(family: str = "all", include_pair_dropout: bool = False,
                     version: str = "v1.1", channel_noise_snr=None) -> list[Condition]:
    """One clean condition plus the requested, centrally defined stress matrix."""
    valid = ("all", "none", "broadband", "powerline", "channel-dropout", "channel-noise")
    if family not in valid:
        raise ValueError(f"unknown corruption family {family!r}")
    if include_pair_dropout and family not in ("all", "channel-dropout"):
        raise ValueError("--include-pair-dropout requires all or channel-dropout")
    config = resolved_corruption_config(version, channel_noise_snr)
    reference = config["power_reference"]
    conditions = [Condition("clean", NoCorruption())]
    if family in ("all", "broadband"):
        conditions += [Condition(f"broadband_snr{snr:g}", GaussianNoise(snr, power_reference=reference), snr)
                       for snr in config["broadband_snr_db"]]
    if family in ("all", "powerline"):
        for amplitude in config["powerline_amplitude_rms"]:
            suffix = f"{amplitude:.2f}" if version == "v1" else f"{amplitude:g}"
            conditions.append(Condition(f"powerline_rms{suffix}".replace(".", "p"),
                                        PowerlineInterference(amplitude, power_reference=reference), amplitude))
    if family in ("all", "channel-dropout"):
        conditions += [Condition(f"dropout_CH{ch + 1}", ChannelDropout((ch,)), channel=f"CH{ch + 1}")
                       for ch in range(CHANNEL_COUNT)]
        if include_pair_dropout:
            conditions += [Condition(f"dropout_CH{a + 1}_CH{b + 1}", ChannelDropout((a, b)),
                                     channel=f"CH{a + 1}|CH{b + 1}")
                           for a, b in combinations(range(CHANNEL_COUNT), 2)]
    if family in ("all", "channel-noise"):
        snrs = config["single_channel_noise_snr_db"]
        if isinstance(snrs, (int, float)):
            snrs = [snrs]
        conditions += [Condition(f"channel_noise_CH{ch + 1}_snr{snr:g}",
                                 GaussianNoise(snr, (ch,), name="single_channel_noise", power_reference=reference),
                                 snr, f"CH{ch + 1}") for snr in snrs for ch in range(CHANNEL_COUNT)]
    return conditions
