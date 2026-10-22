"""Review sample counts, saturation, and unfiltered activity in a recorded session."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CHANNELS = [f"CH{channel}" for channel in range(8)]
SIGNED_24_LIMIT = 2**23


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = json.loads((args.session / "metadata.json").read_text(encoding="utf-8"))
    with (args.session / "samples.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("session contains no samples")

    values = np.asarray(
        [[float(row[channel]) for channel in CHANNELS] for row in rows],
        dtype=np.float64,
    )
    baseline_mask = np.asarray([row["phase"] == "baseline" for row in rows])
    if not np.any(baseline_mask):
        raise ValueError("session has no baseline samples")
    baseline = values[baseline_mask]
    baseline_rms = _demeaned_rms(baseline)

    print(f"Session: {args.session}")
    print(f"Source unit: {metadata.get('value_unit', 'unknown')}")
    print(f"Samples: {len(rows)} at nominal {metadata['sample_rate_hz']} Hz")
    if metadata.get("label_timing_valid") is False:
        print("WARNING: label timing is marked invalid; do not train on this session.")
        if metadata.get("known_issue"):
            print(f"Reason: {metadata['known_issue']}")
    if metadata.get("session_complete") is False:
        print("WARNING: session is incomplete; do not train on it.")
        if metadata.get("known_issue"):
            print(f"Reason: {metadata['known_issue']}")
    print("\nRaw channel check (baseline is demeaned, before filtering):")
    print("channel       minimum       maximum  baseline_RMS  near_rail")
    near_rail = np.abs(values) >= 0.99 * SIGNED_24_LIMIT
    for channel in range(8):
        print(
            f"CH{channel:<2}    {values[:, channel].min():12.0f}  "
            f"{values[:, channel].max():12.0f}  "
            f"{baseline_rms[channel]:12.1f}  "
            f"{100.0 * near_rail[:, channel].mean():8.3f}%"
        )

    gestures = sorted(
        {row["gesture"] for row in rows if row["phase"] == "gesture"}
    )
    print("\nGesture RMS / baseline RMS (screening metric, not a model score):")
    print("gesture       " + "  ".join(f"CH{channel}" for channel in range(8)))
    largest_ratio = 0.0
    for gesture in gestures:
        mask = np.asarray(
            [row["phase"] == "gesture" and row["gesture"] == gesture for row in rows]
        )
        ratios = np.divide(
            _demeaned_rms(values[mask]),
            baseline_rms,
            out=np.full(8, np.inf),
            where=baseline_rms > 0,
        )
        largest_ratio = max(largest_ratio, float(np.max(ratios)))
        print(f"{gesture:<12} " + "  ".join(f"{ratio:5.2f}" for ratio in ratios))

    if np.any(near_rail):
        print("\nWARNING: at least one channel approached the signed 24-bit rail.")
    else:
        print("\nNo samples approached the provisional signed 24-bit rails.")
    if largest_ratio >= 1.5:
        print("At least one gesture/channel shows clearly elevated raw activity.")
    else:
        print("WARNING: raw gesture activity is not clearly above the initial baseline.")
    return 0


def _demeaned_rms(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    return np.sqrt(np.mean(centered**2, axis=0))


if __name__ == "__main__":
    raise SystemExit(main())
