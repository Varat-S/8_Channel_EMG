"""Create a separate four-class P3 deployment candidate and its audit report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib
import numpy as np

from emg.compact_model import make_dataset, read_recording
from emg.dataset_v2 import FIVE_CORE_GESTURES
from emg.model import load_model, save_model
from evaluate_five_core_reapplication import (
    independent_rest_indices,
    indices_for_trials,
    new_model,
    preregistered_trial_split,
    score,
)


CLASSES = ("rest", "fist", "splayed", "wrist_up")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-model",
        type=Path,
        default=Path("models/compact_A_20261006_five_core_shrinkage_v2.joblib"),
    )
    parser.add_argument(
        "--session", type=Path, default=Path("recordings/S01/five_core_reapply_P3")
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/four_core_P3_20261010_shrinkage_v1.joblib"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/four_core_P3_candidate_v1")
    )
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    output = args.output.resolve()
    model_output = args.model_output.resolve()
    if output.exists() or model_output.exists():
        raise ValueError("use fresh result and model paths; existing artifacts are preserved")

    source_artifact = load_model(args.source_model)
    profile = source_artifact["compact_profile"]
    recording = read_recording(
        args.session,
        purpose="held_out_evaluation",
        expected_gestures=FIVE_CORE_GESTURES,
    )
    dataset, _, scale = make_dataset(
        [recording], notch_hz=profile["notch_hz"], normalization="session_baseline"
    )
    calibration_by_class, evaluation_by_class = preregistered_trial_split(
        dataset, args.seed, all_trials=recording.metadata["trials"]
    )
    calibration_ids = sum((calibration_by_class[label][:2] for label in CLASSES), [])
    evaluation_ids = sum((evaluation_by_class[label] for label in CLASSES), [])
    calibration = indices_for_trials(dataset, calibration_ids)
    evaluation = indices_for_trials(dataset, evaluation_ids)
    rest = independent_rest_indices(dataset)

    evaluation_model = new_model(len(CLASSES)).fit(
        dataset.data.features[calibration], dataset.data.labels[calibration]
    )
    evaluated = score(
        evaluation_model,
        dataset,
        evaluation,
        rest,
        strategy="four_core_P3_ablation",
        reps=2,
        normalization="P3_baseline",
    )

    deployment = np.asarray(
        [
            index
            for index, window in enumerate(dataset.windows)
            if window.phase == "gesture" and dataset.data.labels[index] in set(CLASSES)
        ],
        dtype=int,
    )
    deployment_model = new_model(len(CLASSES)).fit(
        dataset.data.features[deployment], dataset.data.labels[deployment]
    )
    save_model(
        model_output,
        deployment_model,
        window_size=25,
        hop_size=5,
        thresholds=0.0,
        sessions=[str(recording.path)],
        bandpass_hz=(10.0, 55.0),
        sample_rate_hz=125.0,
        spatial_reference="common_average",
        baseline_normalize=False,
    )
    artifact = joblib.load(model_output)
    artifact["compact_profile"] = {
        "version": 1,
        "notch_hz": profile["notch_hz"],
        "notch_q": profile["notch_q"],
        "normalization": "fixed_reference",
        "fixed_reference_scale": scale.tolist(),
        "filter_reset_guard_samples": profile["filter_reset_guard_samples"],
        "smoothing_windows": profile["smoothing_windows"],
    }
    artifact["rejection_profile"] = {
        "version": 1,
        "policy": "hold",
        "minimum_output_confidence": 0.0,
        "selection_protocol": "none; four-class probabilities require separate live validation",
    }
    artifact["model_description"] = (
        "Four-class P3-adapted shrinkage-LDA: rest, fist, splayed, and wrist-up. "
        "Wrist-down is intentionally excluded."
    )
    artifact["training_provenance"] = {
        "status": "development_candidate_requires_P4_confirmation",
        "experiment": "four_core_P3_candidate_v1",
        "class_set": list(CLASSES),
        "source": recording.audit,
        "deployment_training_trial_ids": sorted(set(dataset.data.groups[deployment])),
        "excluded_from_model": ["wrist_down", "P3 wrist-down focus block"],
        "P3_was_used_for_model_choice": True,
    }
    joblib.dump(artifact, model_output)

    output.mkdir(parents=True)
    summary = {
        "status": "development candidate; later P4 confirmation required",
        "model": str(model_output),
        "model_sha256": sha256(model_output),
        "classes": list(CLASSES),
        "classifier": "StandardScaler + shrinkage LDA",
        "deployment_trials": len(set(dataset.data.groups[deployment])),
        "deployment_windows": len(deployment),
        "development_evaluation": evaluated[0],
        "development_per_class": evaluated[1]["classification"]["per_class"],
        "training_note": (
            "Deployment refit uses all eligible P3 trials of the four retained classes; "
            "reported metrics come from the earlier fixed 2-train/3-test split."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    run_config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "source_model": str(args.source_model.resolve()),
        "source_model_sha256": sha256(args.source_model),
        "session_audit": recording.audit,
        "calibration_trial_ids": calibration_ids,
        "fixed_evaluation_trial_ids_by_class": {
            label: evaluation_by_class[label] for label in CLASSES
        },
        "deployment_trial_ids": sorted(set(dataset.data.groups[deployment])),
        "source_sha256": sha256(Path(__file__)),
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    (output / "evaluation_details.json").write_text(
        json.dumps(evaluated[1], indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, KeyError, OSError, TypeError) as error:
        raise SystemExit(f"Four-core training error: {error}")
