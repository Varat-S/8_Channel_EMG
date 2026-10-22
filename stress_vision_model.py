"""Stress the frozen CNN on its source session; diagnostic, not external validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_vision_posture import load_session, score
from emg.cnn import SmallEmgCnn


def run(session: Path, model_path: Path) -> list[dict]:
    data = load_session(session)
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    model = SmallEmgCnn(len(artifact["classes"]))
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    labels = tuple(str(x) for x in artifact["classes"])
    mean = np.asarray(artifact["channel_mean"])
    std = np.asarray(artifact["channel_std"])
    windows = data["windows"]
    rng = np.random.default_rng(7)
    conditions = {"clean": windows}
    for channel in range(8):
        dropped = windows.copy()
        dropped[:, channel, :] = 0
        conditions[f"drop_ch{channel + 1}"] = dropped
        weak = windows.copy()
        weak[:, channel, :] *= 0.5
        conditions[f"half_gain_ch{channel + 1}"] = weak
    for gain in (0.5, 1.5):
        conditions[f"all_gain_{gain}"] = windows * gain
    for snr in (20, 10):
        rms = np.sqrt(np.mean(windows**2, axis=(0, 2), keepdims=True))
        conditions[f"noise_{snr}db"] = windows + rng.normal(size=windows.shape).astype(np.float32) * rms * 10 ** (-snr / 20)
    for offset in (1, 2):
        # Small within-window acquisition timing shifts, with replicated edge samples.
        shifted = windows.copy()
        shifted[:, :, offset:] = windows[:, :, :-offset]
        shifted[:, :, :offset] = windows[:, :, :1]
        conditions[f"timing_lag_{offset}_samples"] = shifted
    output = []
    for name, altered in conditions.items():
        with torch.no_grad():
            probabilities = torch.softmax(
                model(torch.tensor((altered - mean) / std, dtype=torch.float32)), dim=1
            ).numpy()
        result = score(labels, data["labels"], data["groups"], probabilities)
        output.append({
            "condition": name,
            "raw_macro_f1": result["raw"]["macro_f1"],
            "smoothed_macro_f1": result["smoothed"]["macro_f1"],
            "smoothed_accuracy": result["smoothed"]["accuracy"],
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=Path("recordings/S01/vision_20261016_192456"))
    parser.add_argument("--model", type=Path, default=Path("models/vision_posture_20261016_192456/cnn_all_postures.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/vision_posture_20261016_192456/stress_diagnostic.csv"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = run(args.session, args.model)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
