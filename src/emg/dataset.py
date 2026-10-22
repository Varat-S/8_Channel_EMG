"""Convert recorded calibration sessions into leakage-safe feature datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .features import extract_time_domain_features
from .preprocessing import apply_spatial_reference, filter_recording
from .types import CHANNEL_COUNT


@dataclass(frozen=True, slots=True)
class FeatureDataset:
    features: NDArray[np.float64]
    labels: NDArray[np.str_]
    groups: NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class WindowDataset:
    windows: NDArray[np.float64]
    labels: NDArray[np.str_]
    groups: NDArray[np.str_]


def load_window_dataset(
    session_directories: list[Path],
    *,
    window_size: int = 25,
    hop_size: int = 5,
    bandpass_hz: tuple[float, float] | None = (10.0, 55.0),
    guard_samples: int = 38,
    spatial_reference: str = "common_average",
    baseline_normalize: bool = True,
) -> WindowDataset:
    windows: list[NDArray[np.float64]] = []
    labels: list[str] = []
    groups: list[str] = []

    for session_directory in session_directories:
        metadata, rows, values, discontinuities = _load_preprocessed_session(
            session_directory,
            bandpass_hz=bandpass_hz,
            spatial_reference=spatial_reference,
            baseline_normalize=baseline_normalize,
        )
        del metadata
        for segment_number, (start, end) in enumerate(
            _segment_bounds(rows, discontinuities)
        ):
            segment_rows = rows[start:end]
            if not segment_rows or segment_rows[0]["phase"] == "prepare":
                continue
            segment = values[start:end]
            if len(segment) <= 2 * guard_samples:
                continue
            segment = segment[guard_samples:-guard_samples]
            group = f"{session_directory.resolve()}::segment_{segment_number}"
            for window_start in range(0, len(segment) - window_size + 1, hop_size):
                # Conv1d expects channel first: (channels, time).
                windows.append(
                    segment[window_start : window_start + window_size].T.copy()
                )
                labels.append(segment_rows[0]["gesture"])
                groups.append(group)

    if not windows:
        raise ValueError("no complete windows were found in the supplied sessions")
    return WindowDataset(
        windows=np.stack(windows),
        labels=np.asarray(labels),
        groups=np.asarray(groups),
    )


def load_feature_dataset(
    session_directories: list[Path],
    *,
    window_size: int = 25,
    hop_size: int = 5,
    thresholds: float | NDArray[np.float64] = 0.0,
    bandpass_hz: tuple[float, float] | None = (10.0, 55.0),
    guard_samples: int = 38,
    spatial_reference: str = "none",
    baseline_normalize: bool = False,
) -> FeatureDataset:
    if window_size < 3 or hop_size <= 0 or guard_samples < 0:
        raise ValueError(
            "window_size must be >= 3; hop_size positive; guard_samples non-negative"
        )

    all_features: list[NDArray[np.float64]] = []
    all_labels: list[str] = []
    all_groups: list[str] = []

    for session_directory in session_directories:
        metadata = json.loads(
            (session_directory / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata.get("label_timing_valid") is False:
            raise ValueError(
                f"session is marked as having invalid label timing: {session_directory}"
            )
        if metadata.get("session_complete") is False:
            raise ValueError(
                f"session is marked incomplete: {session_directory}"
            )
        rows = _read_rows(session_directory / "samples.csv")
        discontinuities = _read_discontinuities(session_directory / "events.csv")
        session_values = np.asarray(
            [
                [float(row[f"CH{channel}"]) for channel in range(CHANNEL_COUNT)]
                for row in rows
            ],
            dtype=np.float64,
        )
        session_values = apply_spatial_reference(session_values, spatial_reference)
        if bandpass_hz is not None:
            session_values = filter_recording(
                session_values,
                sample_rate_hz=float(metadata["sample_rate_hz"]),
                low_hz=bandpass_hz[0],
                high_hz=bandpass_hz[1],
            )
        if baseline_normalize:
            baseline_mask = np.asarray(
                [row["phase"] == "baseline" for row in rows]
            )
            baseline_rms = np.sqrt(
                np.mean(session_values[baseline_mask] ** 2, axis=0)
            )
            if np.any(baseline_rms <= np.finfo(np.float64).eps):
                raise ValueError(f"zero baseline RMS in session: {session_directory}")
            session_values = session_values / baseline_rms
        segment_number = 0
        segment_start = 0

        for index in range(1, len(rows) + 1):
            boundary = (
                index == len(rows)
                or int(rows[index]["sample_index"]) in discontinuities
                or _label_key(rows[index]) != _label_key(rows[index - 1])
            )
            if not boundary:
                continue

            segment = rows[segment_start:index]
            if segment and segment[0]["phase"] != "prepare":
                label = segment[0]["gesture"]
                samples = session_values[segment_start:index]
                if len(samples) > 2 * guard_samples:
                    samples = samples[guard_samples:-guard_samples]
                else:
                    samples = samples[0:0]
                group = f"{session_directory.resolve()}::segment_{segment_number}"
                for start in range(0, len(samples) - window_size + 1, hop_size):
                    all_features.append(
                        extract_time_domain_features(
                            samples[start : start + window_size], thresholds
                        )
                    )
                    all_labels.append(label)
                    all_groups.append(group)

            segment_number += 1
            segment_start = index

    if not all_features:
        raise ValueError("no complete windows were found in the supplied sessions")

    return FeatureDataset(
        features=np.vstack(all_features),
        labels=np.asarray(all_labels),
        groups=np.asarray(all_groups),
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_discontinuities(path: Path) -> set[int]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return {
            int(row["sample_index"])
            for row in csv.DictReader(file)
            if row["event"] == "stream_reconnect"
        }


def _load_preprocessed_session(
    session_directory: Path,
    *,
    bandpass_hz: tuple[float, float] | None,
    spatial_reference: str,
    baseline_normalize: bool,
) -> tuple[dict, list[dict[str, str]], NDArray[np.float64], set[int]]:
    metadata = json.loads(
        (session_directory / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("label_timing_valid") is False:
        raise ValueError(
            f"session is marked as having invalid label timing: {session_directory}"
        )
    if metadata.get("session_complete") is False:
        raise ValueError(f"session is marked incomplete: {session_directory}")
    rows = _read_rows(session_directory / "samples.csv")
    discontinuities = _read_discontinuities(session_directory / "events.csv")
    values = np.asarray(
        [
            [float(row[f"CH{channel}"]) for channel in range(CHANNEL_COUNT)]
            for row in rows
        ],
        dtype=np.float64,
    )
    values = apply_spatial_reference(values, spatial_reference)
    if bandpass_hz is not None:
        values = filter_recording(
            values,
            sample_rate_hz=float(metadata["sample_rate_hz"]),
            low_hz=bandpass_hz[0],
            high_hz=bandpass_hz[1],
        )
    if baseline_normalize:
        baseline_mask = np.asarray([row["phase"] == "baseline" for row in rows])
        baseline_rms = np.sqrt(np.mean(values[baseline_mask] ** 2, axis=0))
        if np.any(baseline_rms <= np.finfo(np.float64).eps):
            raise ValueError(f"zero baseline RMS in session: {session_directory}")
        values = values / baseline_rms
    return metadata, rows, values, discontinuities


def _segment_bounds(
    rows: list[dict[str, str]], discontinuities: set[int]
) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(rows) + 1):
        boundary = (
            index == len(rows)
            or int(rows[index]["sample_index"]) in discontinuities
            or _label_key(rows[index]) != _label_key(rows[index - 1])
        )
        if boundary:
            bounds.append((start, index))
            start = index
    return bounds


def _label_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["phase"], row["gesture"], row["trial"]
