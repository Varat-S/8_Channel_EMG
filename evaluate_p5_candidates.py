"""Prospective P5 scoring for pooled P3/P4 CNN candidates and labeled calibration."""

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
from emg.vision_reapplication import load_protocol, sha256
from evaluate_vision_posture import score
from evaluate_vision_reapplication import (cnn_embeddings, cnn_probabilities,
                                           load_target, normalize_windows)


VARIANTS = ("plain", "gain", "gain_dropout")


def verify_manifest(models: Path, protocol: dict) -> dict:
    manifest = json.loads((models / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "frozen before P5 collection"
            or manifest.get("primary_variant") != "plain"
            or manifest.get("secondary_variants") != ["gain", "gain_dropout"]
            or manifest.get("p5_protocol_sha256") != protocol["_sha256"]):
        raise ValueError("frozen P5 model manifest mismatch")
    for variant in VARIANTS:
        if sha256(models / f"cnn_{variant}.pt") != manifest["model_sha256"].get(variant):
            raise ValueError(f"frozen model hash mismatch: {variant}")
    return manifest


def load_candidate(path: Path, expected_sessions: tuple[str, str], labels: tuple[str, ...]):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(artifact.get("training_sessions", ())) != expected_sessions:
        raise ValueError(f"unexpected training sessions in {path}")
    if tuple(artifact.get("classes", ())) != labels:
        raise ValueError(f"class mismatch in {path}")
    model = SmallEmgCnn(len(labels))
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, artifact


def evaluate(session: Path, models: Path, protocol: dict, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest = verify_manifest(models, protocol)
    data = load_target(session, protocol)
    target_id = data["metadata"]["session_id"]
    labels = tuple(sorted(protocol["gestures"]))
    test = np.isin(data["repetitions"], protocol["test_design"]["fixed_test_repetitions"])
    calibration = np.isin(data["repetitions"], protocol["test_design"]["calibration_repetitions"])
    if len(set(data["groups"][test])) != 42 or len(set(data["groups"][calibration])) != 42:
        raise ValueError("expected 42 calibration and 42 fixed test trials")
    if set(data["groups"][test]) & set(data["groups"][calibration]):
        raise ValueError("calibration/test trial leakage")
    windows = normalize_windows(data, data["fresh_scales"])
    expected_sessions = tuple(manifest["training_sessions"])
    if expected_sessions != ("vision_20261016_192456", "vision_20261022_202644"):
        raise ValueError("unexpected frozen training sessions")
    if target_id in expected_sessions:
        raise ValueError("target application was used to train the pooled models")
    zero_shot = {}
    loaded = {}
    for variant in VARIANTS:
        model, artifact = load_candidate(models / f"cnn_{variant}.pt", expected_sessions, labels)
        loaded[variant] = (model, artifact)
        probabilities = cnn_probabilities(model, artifact, windows[test])
        zero_shot[variant] = score(labels, data["labels"][test], data["groups"][test], probabilities)
    output.mkdir(parents=True)
    first = {"scope": "prospective P5; no P5 gesture labels used in pooled model training",
             "protocol_sha256": protocol["_sha256"], "session": str(session.resolve()),
             "training_sessions": expected_sessions,
             "model_sha256": manifest["model_sha256"],
             "primary_model": "plain", "secondary_models": ["gain", "gain_dropout"],
             "baseline_strategy": "fresh rest baseline per posture, no gesture-label adaptation",
             "fixed_test_trials": 42, "zero_shot": zero_shot,
             "signal_quality": data["signal_quality"]}
    (output / "zero_shot.json").write_text(json.dumps(first, indent=2) + "\n", encoding="utf-8")
    print("Pooled zero-shot scores saved before calibration fitting.", flush=True)
    model, artifact = loaded["plain"]
    embeddings = cnn_embeddings(model, artifact, windows)
    curve = []
    for count in (1, 2, 3):
        train = calibration & (data["repetitions"] <= count)
        train_groups = set(data["groups"][train])
        if len(train_groups) != 14 * count or train_groups & set(data["groups"][test]):
            raise ValueError("invalid nested calibration split")
        classifier = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=np.full(len(labels), 1 / len(labels))))
        classifier.fit(embeddings[train], data["labels"][train])
        if tuple(classifier.classes_) != labels:
            raise ValueError("calibration class order mismatch")
        probabilities = classifier.predict_proba(embeddings[test])
        curve.append({"calibration_trials_per_gesture_per_posture": count,
                      "calibration_trials": len(train_groups),
                      "metrics": score(labels, data["labels"][test], data["groups"][test], probabilities)})
    report = {**first, "calibration_method": "new-only shrinkage LDA on frozen pooled-CNN embeddings",
              "calibration_curve": curve}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("stage", "variant", "calibration_trials",
                                                  "smoothed_accuracy", "smoothed_macro_f1"))
        writer.writeheader()
        for variant, metrics in zero_shot.items():
            writer.writerow({"stage": "zero_shot", "variant": variant, "calibration_trials": 0,
                             "smoothed_accuracy": metrics["smoothed"]["accuracy"],
                             "smoothed_macro_f1": metrics["smoothed"]["macro_f1"]})
        for row in curve:
            metrics = row["metrics"]["smoothed"]
            writer.writerow({"stage": "calibration", "variant": "plain",
                             "calibration_trials": row["calibration_trials"],
                             "smoothed_accuracy": metrics["accuracy"],
                             "smoothed_macro_f1": metrics["macro_f1"]})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models/vision_multisession_P3_P4_v1"))
    parser.add_argument("--protocol", type=Path, default=Path("data/vision_reapplication_protocol_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol, Path(__file__).resolve().parent)
    report = evaluate(args.session, args.models, protocol, args.output)
    print((args.output / "summary.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
