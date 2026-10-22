"""Replay a saved vision session through the frozen seven gesture CNN."""

from __future__ import annotations

import argparse
import csv
from collections import deque
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch

from emg.cnn import SmallEmgCnn
from emg.evaluation.metrics import calculate_metrics
from emg.preprocessing import apply_spatial_reference, filter_recording


def replay(session: Path, artifact_path: Path, *, confidence_floor: float = 0.55) -> dict:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("session_complete"):
        raise ValueError("replay requires a completed session")
    with (session / "samples.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    raw = np.array([[float(row[f"CH{i}"]) for i in range(8)] for row in rows])
    if not np.all(np.isfinite(raw)):
        raise ValueError("nonfinite EMG values")
    referenced = apply_spatial_reference(raw, artifact["spatial_reference"])
    filtered = filter_recording(
        referenced,
        sample_rate_hz=125,
        low_hz=artifact["bandpass_hz"][0],
        high_hz=artifact["bandpass_hz"][1],
    )
    baseline = np.array([row["phase"] == "baseline" for row in rows])
    if not baseline.any():
        raise ValueError("missing session baseline")
    scale = np.sqrt(np.mean(filtered[baseline] ** 2, axis=0))
    if np.any(scale < 1e-8):
        raise ValueError("flat baseline channel")
    signal = filtered / scale
    model = SmallEmgCnn(len(artifact["classes"]))
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    mean = np.asarray(artifact["channel_mean"])
    std = np.asarray(artifact["channel_std"])
    size, hop, guard = (int(artifact[key]) for key in ("window_size", "hop_size", "guard_samples"))
    labels = tuple(str(label) for label in artifact["classes"])
    truths, predictions, groups, confidences = [], [], [], []
    per_trial = []
    for trial in metadata["trials"]:
        indices = np.array([
            i for i, row in enumerate(rows)
            if row["phase"] == "gesture" and row["trial"] == trial["trial_id"]
        ], dtype=int)
        indices = indices[guard:-guard]
        windows = np.stack([
            signal[indices[start:start + size]].T
            for start in range(0, len(indices) - size + 1, hop)
        ])
        with torch.no_grad():
            probabilities = torch.softmax(
                model(torch.tensor((windows - mean) / std, dtype=torch.float32)), dim=1
            ).numpy()
        history = deque(maxlen=7)
        trial_predictions = []
        for probability in probabilities:
            history.append(probability)
            averaged = np.mean(history, axis=0)
            confidence = float(np.max(averaged))
            prediction = labels[int(np.argmax(averaged))] if confidence >= confidence_floor else "uncertain"
            trial_predictions.append(prediction)
            truths.append(trial["gesture"])
            predictions.append(prediction)
            groups.append(trial["trial_id"])
            confidences.append(confidence)
        per_trial.append({
            "trial_id": trial["trial_id"],
            "truth": trial["gesture"],
            "prediction": max(set(trial_predictions), key=trial_predictions.count),
            "uncertain_fraction": float(np.mean(np.array(trial_predictions) == "uncertain")),
        })
    accepted = np.array([prediction != "uncertain" for prediction in predictions])
    truth = np.array(truths)[accepted]
    predicted = np.array(predictions)[accepted]
    selected_groups = np.array(groups)[accepted]
    metrics = calculate_metrics(truth, predicted, selected_groups, labels, hop / 125)
    return {
        "scope": "offline training-session replay; diagnostic only, not prospective accuracy",
        "session": str(session),
        "artifact": str(artifact_path),
        "confidence_floor": confidence_floor,
        "windows": len(truths),
        "coverage": float(np.mean(accepted)),
        "accepted_window_metrics": metrics,
        "per_trial": per_trial,
        "signal_quality": {
            "nonfinite_values": int(np.count_nonzero(~np.isfinite(raw))),
            "clipped_fraction_at_24_bit_limit": float(np.mean(np.abs(raw) >= 8388607)),
            "baseline_rms_after_filter": scale.tolist(),
            "raw_channel_std": raw.std(axis=0).tolist(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=Path("recordings/S01/vision_20261016_192456"))
    parser.add_argument("--model", type=Path, default=Path("models/vision_posture_20261016_192456/cnn_all_postures.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/vision_posture_20261016_192456/replay_diagnostic.json"))
    parser.add_argument("--confidence-floor", type=float, default=0.55)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = replay(args.session, args.model, confidence_floor=args.confidence_floor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("windows", "coverage", "signal_quality")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
