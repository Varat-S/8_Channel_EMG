"""Pilot one/two-trial-per-posture adaptation between P3 and P4 applications."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emg.cnn import SmallEmgCnn
from evaluate_vision_posture import load_session, score
from evaluate_vision_reapplication import cnn_embeddings, cnn_probabilities


TEST_REPETITIONS = (3, 6)
CALIBRATION_REPETITIONS = {1: (1, 4), 2: (1, 2, 4, 5)}


def run_direction(source_path: Path, target_path: Path, model_path: Path) -> dict:
    source = load_session(source_path)
    target = load_session(target_path)
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    labels = tuple(artifact["classes"])
    if artifact.get("training_session") != source["metadata"]["session_id"]:
        raise ValueError("frozen model does not belong to source session")
    if tuple(sorted(set(target["labels"]))) != labels:
        raise ValueError("target gesture labels do not match model")
    model = SmallEmgCnn(len(labels))
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    test = np.isin(target["repetitions"], TEST_REPETITIONS)
    if len(set(target["groups"][test])) != 14:
        raise ValueError("expected 14 fixed test trials")
    result = {"source_session": source["metadata"]["session_id"],
              "target_session": target["metadata"]["session_id"], "normalizations": {}}
    scales = {"strict_source_baseline": target["baseline_scale"] / source["baseline_scale"],
              "fresh_target_baseline": np.ones(8)}
    for normalization, ratio in scales.items():
        windows = (target["windows"] * ratio[None, :, None]).astype(np.float32)
        frozen = cnn_probabilities(model, artifact, windows[test])
        methods = {"zero_shot": {"calibration_trials": 0,
                                 "metrics": score(labels, target["labels"][test],
                                                  target["groups"][test], frozen)}}
        embeddings = cnn_embeddings(model, artifact, windows)
        for count, repetitions in CALIBRATION_REPETITIONS.items():
            calibration = np.isin(target["repetitions"], repetitions)
            train_groups = set(target["groups"][calibration])
            test_groups = set(target["groups"][test])
            if train_groups & test_groups or len(train_groups) != 14 * count:
                raise ValueError("calibration/test trial split is invalid")
            classifier = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto", priors=np.full(len(labels), 1 / len(labels))))
            classifier.fit(embeddings[calibration], target["labels"][calibration])
            if tuple(classifier.classes_) != labels:
                raise ValueError("calibration class order mismatch")
            probabilities = classifier.predict_proba(embeddings[test])
            methods[f"calibration_{count}_per_posture"] = {
                "calibration_trials": len(train_groups),
                "metrics": score(labels, target["labels"][test], target["groups"][test], probabilities),
            }
        result["normalizations"][normalization] = methods
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--p4", type=Path, required=True)
    parser.add_argument("--p3-model", type=Path, required=True)
    parser.add_argument("--p4-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {
        "scope": "exploratory pilot on two known applications; not a future-application accuracy estimate",
        "fixed_test_repetitions": TEST_REPETITIONS,
        "calibration_repetitions": CALIBRATION_REPETITIONS,
        "method": "new-only shrinkage LDA on frozen 32-dimensional CNN embeddings",
        "directions": [run_direction(args.p3, args.p4, args.p3_model),
                       run_direction(args.p4, args.p3, args.p4_model)],
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("source", "target", "normalization", "method",
                                                  "calibration_trials", "test_trials", "smoothed_accuracy",
                                                  "smoothed_macro_f1"))
        writer.writeheader()
        for direction in report["directions"]:
            for normalization, methods in direction["normalizations"].items():
                for method, row in methods.items():
                    metrics = row["metrics"]["smoothed"]
                    writer.writerow({"source": direction["source_session"],
                                     "target": direction["target_session"],
                                     "normalization": normalization, "method": method,
                                     "calibration_trials": row["calibration_trials"],
                                     "test_trials": 14,
                                     "smoothed_accuracy": metrics["accuracy"],
                                     "smoothed_macro_f1": metrics["macro_f1"]})
    print((args.output / "summary.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
