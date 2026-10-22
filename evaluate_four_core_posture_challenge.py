"""Score a held-out four-core posture challenge without fitting or tuning anything."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from emg.compact_model import make_dataset, read_recording
from emg.evaluation.benchmark import (
    DEFAULT_CONFIG,
    _csv,
    _json,
    _plot_confusion,
    evaluate_model,
    prediction_rows,
)
from emg.evaluation.dataset_v2 import independent_rest_metrics
from emg.evaluation.metrics import calculate_metrics
from emg.model import load_model


CLASSES = ("rest", "fist", "splayed", "wrist_up")
DEFAULT_MODEL = Path("models/four_core_P3_20261010_shrinkage_v1.joblib")
DEFAULT_SESSION = Path("recordings/S01/four_core_P3_posture_challenge_20261010")
DEFAULT_OUTPUT = Path("results/four_core_P3_posture_challenge_v1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indices(dataset, phases):
    phases = set(phases)
    return np.asarray(
        [i for i, window in enumerate(dataset.windows) if window.phase in phases],
        dtype=int,
    )


def score_dataset(model, dataset, conditions, posture_order):
    gesture = indices(dataset, ("gesture",))
    independent_rest = indices(dataset, ("rest_pre", "rest_post"))
    combined = np.sort(np.concatenate((gesture, independent_rest)))
    result = evaluate_model(model, dataset, combined, DEFAULT_CONFIG)
    is_gesture = np.asarray(
        [dataset.windows[i].phase == "gesture" for i in result.test_indices]
    )
    gesture_result_indices = result.test_indices[is_gesture]
    gesture_groups = np.asarray(
        [dataset.windows[i].group_id for i in gesture_result_indices]
    )
    overall = calculate_metrics(
        dataset.data.labels[gesture_result_indices],
        result.smoothed_predictions[is_gesture],
        gesture_groups,
        model.classes_,
        DEFAULT_CONFIG["hop_size"] / DEFAULT_CONFIG["sample_rate_hz"],
    )
    rest = independent_rest_metrics(
        result.smoothed_predictions[~is_gesture],
        [dataset.windows[i] for i in result.test_indices[~is_gesture]],
        DEFAULT_CONFIG["hop_size"] / DEFAULT_CONFIG["sample_rate_hz"],
    )

    posture_rows = []
    for posture in posture_order:
        selected = np.asarray(
            [i for i in gesture if conditions[i]["arm_posture"] == posture], dtype=int
        )
        posture_result = evaluate_model(model, dataset, selected, DEFAULT_CONFIG)
        metric = posture_result.metrics["smoothed"]
        posture_rows.append(
            {
                "arm_posture": posture,
                "trials": len(set(dataset.data.groups[selected])),
                "windows": len(selected),
                "accuracy": metric["accuracy"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "macro_f1": metric["macro_f1"],
                "rest_false_active_fraction": metric["false_active_window_fraction"],
                "switches_per_second": metric["switches_per_second"],
            }
        )

    lookup = {int(data_index): row for row, data_index in enumerate(result.test_indices)}
    classes = [str(value) for value in model.classes_]
    trial_rows = []
    for trial_id in dict.fromkeys(dataset.data.groups[gesture].tolist()):
        selected = [i for i in gesture if dataset.windows[i].trial_id == trial_id]
        positions = [lookup[int(i)] for i in selected]
        predictions = [str(result.smoothed_predictions[position]) for position in positions]
        counts = Counter(predictions)
        mean_probabilities = np.mean(result.smoothed_probabilities[positions], axis=0)
        winner = max(
            classes,
            key=lambda label: (counts[label], mean_probabilities[classes.index(label)]),
        )
        true_label = str(dataset.data.labels[selected[0]])
        trial_rows.append(
            {
                "trial_id": trial_id,
                "arm_posture": conditions[selected[0]]["arm_posture"],
                "true_label": true_label,
                "majority_prediction": winner,
                "correct": winner == true_label,
                "majority_fraction": counts[winner] / len(predictions),
                "mean_smoothed_confidence": float(
                    np.mean(np.max(result.smoothed_probabilities[positions], axis=1))
                ),
                "windows": len(selected),
            }
        )

    prediction_output = prediction_rows(result, dataset)
    for row, data_index in zip(prediction_output, result.test_indices):
        row.update(conditions[int(data_index)])
        row["evaluation_scope"] = (
            "gesture" if dataset.windows[int(data_index)].phase == "gesture" else "independent_rest"
        )
    return overall, rest, posture_rows, trial_rows, prediction_output


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError("use a fresh output directory; held-out evidence is preserved")
    artifact = load_model(args.model)
    model = artifact["model"]
    if set(str(value) for value in model.classes_) != set(CLASSES):
        raise ValueError("expected the frozen four-core model")
    profile = artifact["compact_profile"]
    if float(artifact["sample_rate_hz"]) != 125 or int(artifact["window_size"]) != 25:
        raise ValueError("unexpected model sampling or window configuration")

    recording = read_recording(
        args.session,
        purpose="held_out_evaluation",
        expected_gestures=CLASSES,
    )
    fixed, fixed_conditions, _ = make_dataset(
        [recording],
        notch_hz=profile["notch_hz"],
        normalization="fixed_reference",
        fixed_scale=profile["fixed_reference_scale"],
    )
    baseline, baseline_conditions, current_scale = make_dataset(
        [recording],
        notch_hz=profile["notch_hz"],
        normalization="session_baseline",
    )
    if not (
        np.array_equal(fixed.data.labels, baseline.data.labels)
        and np.array_equal(fixed.data.groups, baseline.data.groups)
        and fixed_conditions == baseline_conditions
    ):
        raise AssertionError("normalization variants are not aligned")

    fixed_score = score_dataset(
        model, fixed, fixed_conditions, recording.metadata["posture_order"]
    )
    baseline_score = score_dataset(
        model, baseline, baseline_conditions, recording.metadata["posture_order"]
    )
    output.mkdir(parents=True)
    fixed_overall, fixed_rest, fixed_postures, fixed_trials, predictions = fixed_score
    baseline_overall, baseline_rest, baseline_postures, baseline_trials, _ = baseline_score

    trial_summary = {
        "correct": int(sum(bool(row["correct"]) for row in fixed_trials)),
        "total": len(fixed_trials),
        "accuracy": float(np.mean([row["correct"] for row in fixed_trials])),
        "by_posture": {
            posture: {
                "correct": int(sum(row["correct"] for row in fixed_trials if row["arm_posture"] == posture)),
                "total": int(sum(row["arm_posture"] == posture for row in fixed_trials)),
            }
            for posture in recording.metadata["posture_order"]
        },
    }
    summary = {
        "status": "held-out same-application posture evaluation; no refitting or tuning",
        "primary": {
            "normalization": "frozen P3 fixed-reference scale (matches live deployment)",
            "window_metrics": fixed_overall,
            "whole_trial_majority": trial_summary,
            "independent_reference_rest": fixed_rest,
            "per_posture": fixed_postures,
        },
        "sensitivity_check": {
            "normalization": "this recording's five-second baseline; diagnostic only",
            "window_metrics": baseline_overall,
            "independent_reference_rest": baseline_rest,
            "per_posture": baseline_postures,
            "whole_trial_majority_accuracy": float(
                np.mean([row["correct"] for row in baseline_trials])
            ),
        },
        "baseline_scale_ratio_current_over_frozen": (
            np.asarray(current_scale) / np.asarray(profile["fixed_reference_scale"])
        ).tolist(),
    }
    run_config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "session": recording.audit,
        "classes": list(CLASSES),
        "processing": {
            "spatial_reference": artifact["spatial_reference"],
            "bandpass_hz": artifact["bandpass_hz"],
            "notch_hz": profile["notch_hz"],
            "window_samples": artifact["window_size"],
            "hop_samples": artifact["hop_size"],
            "smoothing_windows": profile["smoothing_windows"],
            "confidence_rejection": artifact.get("rejection_profile"),
        },
        "data_policy": "all 36 posture-challenge trials scored once; no training, threshold selection, exclusion, or model replacement",
        "source_sha256": sha256(Path(__file__)),
    }

    _json(output / "summary.json", summary)
    _json(output / "run_config.json", run_config)
    _csv(output / "per_posture.csv", list(fixed_postures[0]), fixed_postures)
    _csv(output / "per_trial.csv", list(fixed_trials[0]), fixed_trials)
    per_class = fixed_overall["per_class"]
    _csv(output / "per_class.csv", list(per_class[0]), per_class)
    _csv(output / "predictions.csv", list(predictions[0]), predictions)
    _plot_confusion(fixed_overall, output / "confusion_matrix.png")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, KeyError, OSError, TypeError) as error:
        raise SystemExit(f"Posture evaluation error: {error}")
