"""Evaluate the frozen seven-gesture models on a new electrode application."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emg.cnn import SmallEmgCnn
from emg.features import extract_time_domain_features
from emg.preprocessing import apply_spatial_reference, filter_recording
from emg.vision_reapplication import load_protocol, schedule, validate_session
from evaluate_vision_posture import (GUARD_SAMPLES, HOP_SIZE, SAMPLE_RATE_HZ,
                                     WINDOW_SIZE, load_session, score, smooth_probabilities)
from benchmark_vision_offline import trial_bootstrap


def load_target(session: Path, protocol: dict, *, allow_synthetic: bool = False) -> dict:
    metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    validate_session(metadata, protocol, allow_synthetic=allow_synthetic)
    with (session / "samples.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    raw = np.asarray([[float(row[f"CH{channel}"]) for channel in range(8)] for row in rows])
    if not np.all(np.isfinite(raw)):
        raise ValueError("nonfinite target channel values")
    values = filter_recording(apply_spatial_reference(raw, "common_average"),
                              sample_rate_hz=SAMPLE_RATE_HZ, low_hz=10, high_hz=55)
    scales = {}
    for baseline in metadata["baselines"]:
        posture = baseline["protocol_posture"]
        indices = np.asarray([i for i, row in enumerate(rows)
                              if row["phase"] == "baseline" and row["trial"] == baseline["trial_id"]])
        if len(indices) != baseline["sample_count"]:
            raise ValueError(f"baseline row count mismatch in {posture}")
        if any(rows[i].get("protocol_posture") != posture for i in indices):
            raise ValueError(f"operator posture column mismatch in {posture} baseline")
        if (np.any(np.diff(indices) != 1)
                or np.any(np.diff([int(rows[i]["sample_index"]) for i in indices]) != 1)
                or int(rows[indices[0]]["sample_index"]) != baseline["start_sample"]
                or int(rows[indices[-1]]["sample_index"]) + 1 != baseline["end_sample"]):
            raise ValueError(f"baseline samples are not contiguous in {posture}")
        scale = np.sqrt(np.mean(values[indices] ** 2, axis=0))
        if np.any(scale < 1e-8) or not np.all(np.isfinite(scale)):
            raise ValueError(f"invalid baseline RMS in {posture}")
        scales[posture] = scale
    windows, labels, groups, postures, repetitions = [], [], [], [], []
    for trial in metadata["trials"]:
        identity = trial["trial_id"]
        indices = np.asarray([i for i, row in enumerate(rows)
                              if row["phase"] == "gesture" and row["trial"] == identity])
        if len(indices) != trial["sample_count"]:
            raise ValueError(f"trial row count mismatch for {identity}")
        if (np.any(np.diff(indices) != 1)
                or np.any(np.diff([int(rows[i]["sample_index"]) for i in indices]) != 1)
                or int(rows[indices[0]]["sample_index"]) != trial["start_sample"]
                or int(rows[indices[-1]]["sample_index"]) + 1 != trial["end_sample"]):
            raise ValueError(f"trial samples are not contiguous for {identity}")
        if any(rows[i].get("protocol_posture") != trial["protocol_posture"] for i in indices):
            raise ValueError(f"operator posture column mismatch for {identity}")
        indices = indices[GUARD_SAMPLES:-GUARD_SAMPLES]
        if len(indices) < WINDOW_SIZE:
            raise ValueError(f"too few guarded samples for {identity}")
        for offset in range(0, len(indices) - WINDOW_SIZE + 1, HOP_SIZE):
            windows.append(values[indices[offset:offset + WINDOW_SIZE]].T)
            labels.append(trial["gesture"])
            groups.append(identity)
            postures.append(trial["protocol_posture"])
            repetitions.append(int(trial["repetition_index"]))
    return {"metadata": metadata, "raw_windows": np.stack(windows).astype(np.float32),
            "labels": np.asarray(labels), "groups": np.asarray(groups),
            "postures": np.asarray(postures), "repetitions": np.asarray(repetitions),
            "fresh_scales": scales,
            "signal_quality": {"raw_channel_std": raw.std(axis=0).tolist(),
                               "clipped_fraction_at_24_bit_limit": float(np.mean(np.abs(raw) >= 8388607)),
                               "baseline_rms_after_filter": {key: value.tolist() for key, value in scales.items()}}}


def normalize_windows(data: dict, scales: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([window / scales[posture][:, None]
                     for window, posture in zip(data["raw_windows"], data["postures"])]).astype(np.float32)


def cnn_probabilities(model, artifact, windows):
    mean = np.asarray(artifact["channel_mean"], dtype=np.float32)
    std = np.asarray(artifact["channel_std"], dtype=np.float32)
    results = []
    model.eval()
    with torch.no_grad():
        for batch in np.array_split(windows, max(1, (len(windows) + 127) // 128)):
            inputs = torch.tensor((batch - mean) / std, dtype=torch.float32)
            results.append(torch.softmax(model(inputs), dim=1).numpy())
    return np.vstack(results)


def cnn_embeddings(model, artifact, windows):
    mean = np.asarray(artifact["channel_mean"], dtype=np.float32)
    std = np.asarray(artifact["channel_std"], dtype=np.float32)
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch in np.array_split(windows, max(1, (len(windows) + 127) // 128)):
            inputs = torch.tensor((batch - mean) / std, dtype=torch.float32)
            outputs.append(model.features(inputs).flatten(1).numpy())
    return np.vstack(outputs)


def lda_probabilities(model, windows, labels):
    features = np.stack([extract_time_domain_features(window.T) for window in windows])
    probabilities = model.predict_proba(features)
    if tuple(str(x) for x in model.classes_) != labels:
        raise ValueError("frozen LDA class order mismatch")
    return probabilities


def trial_details(labels, truth, groups, probabilities):
    smoothed = smooth_probabilities(probabilities, groups)
    predicted = np.asarray(labels)[np.argmax(smoothed, axis=1)]
    details = []
    for group in dict.fromkeys(groups.tolist()):
        mask = groups == group
        trial_predictions = predicted[mask]
        correct = np.flatnonzero(trial_predictions == truth[mask][0])
        details.append({"trial_id": str(group), "truth": str(truth[mask][0]),
                        "window_count": int(mask.sum()),
                        "correct_window_fraction": float(np.mean(trial_predictions == truth[mask])),
                        "first_correct_delay_seconds_from_trial_start":
                            round((GUARD_SAMPLES + WINDOW_SIZE + int(correct[0]) * HOP_SIZE) / SAMPLE_RATE_HZ, 3)
                            if len(correct) else None})
    return details


def evaluate(session: Path, protocol: dict, *, allow_synthetic: bool = False):
    data = load_target(session, protocol, allow_synthetic=allow_synthetic)
    source = ROOT / "recordings/S01/vision_20261016_192456"
    source_scale = load_session(source)["baseline_scale"]
    labels = tuple(sorted(protocol["gestures"]))
    artifact = torch.load(ROOT / protocol["models"]["primary"], map_location="cpu", weights_only=False)
    if tuple(str(label) for label in artifact["classes"]) != labels:
        raise ValueError("frozen CNN class order mismatch")
    model = SmallEmgCnn(len(labels))
    model.load_state_dict(artifact["state_dict"])
    lda = joblib.load(ROOT / protocol["models"]["reference"])["model"]
    test = np.isin(data["repetitions"], protocol["test_design"]["fixed_test_repetitions"])
    calibration = np.isin(data["repetitions"], protocol["test_design"]["calibration_repetitions"])
    if set(data["groups"][test]) & set(data["groups"][calibration]):
        raise ValueError("calibration/test trial leakage")
    strict = normalize_windows(data, {posture: source_scale for posture in data["fresh_scales"]})
    fresh = normalize_windows(data, data["fresh_scales"])
    zero_shot = []
    for method, windows in (("strict_original_baseline", strict), ("fresh_posture_baselines", fresh)):
        for model_name, probabilities in (("frozen_cnn", cnn_probabilities(model, artifact, windows[test])),
                                          ("frozen_lda", lda_probabilities(lda, windows[test], labels))):
            metrics = score(labels, data["labels"][test], data["groups"][test], probabilities)
            interval = None
            if method == "strict_original_baseline" and model_name == "frozen_cnn":
                predicted = np.asarray(labels)[np.argmax(
                    smooth_probabilities(probabilities, data["groups"][test]), axis=1)]
                interval = trial_bootstrap(data["labels"][test], predicted,
                                           data["groups"][test], labels,
                                           postures=data["postures"][test])
            zero_shot.append({"method": method, "model": model_name,
                              "metrics": metrics, "trial_bootstrap_95": interval,
                              "per_trial": trial_details(labels, data["labels"][test],
                                                                               data["groups"][test], probabilities)})
    return data, model, artifact, labels, test, calibration, fresh, zero_shot


def adaptation_curve(data, model, artifact, labels, test, calibration, windows):
    """Fixed new-only shrinkage LDA on frozen 32D CNN embeddings."""
    embeddings = cnn_embeddings(model, artifact, windows)
    rows = []
    for count in (1, 2, 3):
        train = calibration & (data["repetitions"] <= count)
        if set(data["groups"][train]) & set(data["groups"][test]):
            raise ValueError("adaptation trial leakage")
        classifier = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=np.full(len(labels), 1 / len(labels))))
        classifier.fit(embeddings[train], data["labels"][train])
        if tuple(str(label) for label in classifier.classes_) != labels:
            raise ValueError("adaptation class order mismatch")
        probabilities = classifier.predict_proba(embeddings[test])
        rows.append({"calibration_trials_per_gesture_per_posture": count,
                     "training_trial_count": int(len(set(data["groups"][train]))),
                     "metrics": score(labels, data["labels"][test], data["groups"][test], probabilities),
                     "per_trial": trial_details(labels, data["labels"][test], data["groups"][test], probabilities)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/vision_reapplication_protocol_v1.json")
    parser.add_argument("--session", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true", help="software validation only")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol, ROOT)
    if args.plan:
        print(json.dumps({"protocol": protocol["protocol"], "plan_sha256": protocol["_sha256"],
                          "steps": schedule(protocol), "adaptation": "new-only shrinkage LDA on frozen CNN embeddings"}, indent=2))
        return 0
    if args.session is None or args.output is None:
        parser.error("--session and --output are required unless --plan is used")
    if args.output.exists():
        raise FileExistsError(args.output)
    data, model, artifact, labels, test, calibration, fresh, zero_shot = evaluate(
        args.session, protocol, allow_synthetic=args.allow_synthetic)
    args.output.mkdir(parents=True)
    zero_shot_report = {"protocol_sha256": protocol["_sha256"], "session": str(args.session),
                        "scope": "synthetic software check" if args.allow_synthetic else "prospective electrode reapplication",
                        "fixed_test_trial_count": int(len(set(data["groups"][test]))),
                        "zero_shot": zero_shot, "signal_quality": data["signal_quality"]}
    (args.output / "zero_shot.json").write_text(json.dumps(zero_shot_report, indent=2) + "\n", encoding="utf-8")
    print("Frozen zero-shot results saved before target fitting.", flush=True)
    curve = adaptation_curve(data, model, artifact, labels, test, calibration, fresh)
    report = {**zero_shot_report, "adaptation_method": "new-only shrinkage LDA on frozen CNN embeddings; fresh posture baselines",
              "calibration_curve": curve,
              "independent_rest_false_activation": None,
              "independent_rest_note": "No independent rest segment is required by the frozen acquisition schedule; test rest trials are included in per-class metrics."}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = [{"stage": "zero_shot", "method": row["method"], "model": row["model"],
             "calibration_trials_per_gesture_per_posture": 0,
             "raw_macro_f1": row["metrics"]["raw"]["macro_f1"],
             "smoothed_macro_f1": row["metrics"]["smoothed"]["macro_f1"]}
            for row in zero_shot]
    rows.extend({"stage": "adaptation", "method": "fresh_posture_baselines", "model": "frozen_cnn_embedding_lda",
                 "calibration_trials_per_gesture_per_posture": row["calibration_trials_per_gesture_per_posture"],
                 "raw_macro_f1": row["metrics"]["raw"]["macro_f1"],
                 "smoothed_macro_f1": row["metrics"]["smoothed"]["macro_f1"]} for row in curve)
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
