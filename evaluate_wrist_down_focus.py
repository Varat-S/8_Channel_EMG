"""Measure the targeted P3 wrist-down calibration without changing deployment."""
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
from emg.model import load_model
from evaluate_five_core_reapplication import (
    independent_rest_indices,
    indices_for_trials,
    new_model,
    preregistered_trial_split,
    score,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/compact_A_20261006_five_core_shrinkage_v2.joblib"),
    )
    parser.add_argument(
        "--p3-session", type=Path, default=Path("recordings/S01/five_core_reapply_P3")
    )
    parser.add_argument(
        "--focus-session",
        type=Path,
        default=Path(
            "recordings/S01/five_core_reapply_P3_wrist_down_calibration_20261010"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/five_core_P3_wrist_down_focus_v1")
    )
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError("use a fresh output directory; focused evaluation is preserved")
    artifact = load_model(args.model)
    classes = tuple(str(value) for value in artifact["model"].classes_)
    if set(classes) != set(FIVE_CORE_GESTURES):
        raise ValueError("model is not the expected five-core classifier")

    p3 = read_recording(
        args.p3_session,
        purpose="held_out_evaluation",
        expected_gestures=FIVE_CORE_GESTURES,
    )
    focus = read_recording(
        args.focus_session,
        purpose="training",
        expected_gestures=("rest", "wrist_down"),
    )
    for field in ("subject_id", "electrode_placement_id", "recording_date", "source"):
        if p3.metadata[field] != focus.metadata[field]:
            raise ValueError(f"P3 and focus recording differ in {field}")

    profile = artifact["compact_profile"]
    p3_data, _, p3_scale = make_dataset(
        [p3], notch_hz=profile["notch_hz"], normalization="session_baseline"
    )
    focus_data, _, focus_scale = make_dataset(
        [focus], notch_hz=profile["notch_hz"], normalization="session_baseline"
    )
    calibration_by_class, evaluation_by_class = preregistered_trial_split(
        p3_data, args.seed, all_trials=p3.metadata["trials"]
    )
    reps = min(map(len, calibration_by_class.values()))
    if reps != 2:
        raise ValueError(f"expected two balanced clean P3 calibration trials, found {reps}")
    calibration_ids = sum(
        (calibration_by_class[label][:reps] for label in FIVE_CORE_GESTURES), []
    )
    evaluation_ids = sum(evaluation_by_class.values(), [])
    calibration = indices_for_trials(p3_data, calibration_ids)
    test = indices_for_trials(p3_data, evaluation_ids)
    rest = independent_rest_indices(p3_data)
    focus_train = np.asarray(
        [index for index, window in enumerate(focus_data.windows) if window.phase == "gesture"],
        dtype=int,
    )

    base_model = new_model(len(classes)).fit(
        p3_data.data.features[calibration], p3_data.data.labels[calibration]
    )
    base = score(
        base_model,
        p3_data,
        test,
        rest,
        strategy="p3_new_only",
        reps=reps,
        normalization="per_session_baseline",
    )
    augmented_features = np.vstack(
        (p3_data.data.features[calibration], focus_data.data.features[focus_train])
    )
    augmented_labels = np.concatenate(
        (p3_data.data.labels[calibration], focus_data.data.labels[focus_train])
    )
    focused_model = new_model(len(classes)).fit(augmented_features, augmented_labels)
    focused = score(
        focused_model,
        p3_data,
        test,
        rest,
        strategy="p3_plus_all_wrist_down_focus",
        reps=reps,
        normalization="per_session_baseline",
    )

    rows = [base[0], focused[0]]
    output.mkdir(parents=True)
    write_csv(output / "summary.csv", rows)
    joblib.dump(
        {
            "status": "development_only_not_deployed",
            "model": focused_model,
            "classes": list(classes),
            "normalization": "five-second baseline from the active electrode application",
            "p3_baseline_scale": p3_scale.tolist(),
            "training_trial_ids": calibration_ids
            + sorted(set(focus_data.data.groups[focus_train])),
        },
        output / "focused_evaluation_model.joblib",
    )
    details = {
        "warning": (
            "Development result, not untouched confirmation: the focus block was collected "
            "after inspecting P3 wrist-down performance. Reserve a later P4 application for confirmation."
        ),
        "baseline": base[1],
        "focused": focused[1],
        "delta_macro_f1": focused[0]["macro_f1"] - base[0]["macro_f1"],
        "delta_balanced_accuracy": focused[0]["balanced_accuracy"]
        - base[0]["balanced_accuracy"],
        "p3_calibration_trial_ids": calibration_ids,
        "fixed_evaluation_trial_ids_by_class": evaluation_by_class,
        "focus_training_trial_ids": sorted(set(focus_data.data.groups[focus_train])),
        "focus_training_windows": len(focus_train),
        "p3_audit": p3.audit,
        "focus_audit": focus.audit,
    }
    (output / "details.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "model_source": str(args.model.resolve()),
        "model_source_sha256": file_sha256(args.model),
        "p3_session": str(args.p3_session.resolve()),
        "focus_session": str(args.focus_session.resolve()),
        "output_model_status": "development_only_not_deployed",
        "selection_rule": "all four focus trials of both recorded classes included; no subset selected on test performance",
        "source_sha256": file_sha256(Path(__file__)),
    }
    (output / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": rows, "details": details}, indent=2), flush=True)
    return rows, details


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, KeyError, OSError, TypeError) as error:
        raise SystemExit(f"Focused evaluation error: {error}")
