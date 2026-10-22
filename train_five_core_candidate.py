"""Train and document a five-class shrinkage-LDA live candidate.

The confidence policy is selected only from leave-one-recorded-position-out
predictions. The previously inspected unfamiliar session is reported separately as
a development check and is never used for fitting or threshold selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from emg.compact_model import make_dataset, read_collection, read_recording
from emg.evaluation.benchmark import DEFAULT_CONFIG, evaluate_model
from emg.evaluation.dataset_v2 import independent_rest_metrics
from emg.evaluation.metrics import calculate_metrics
from emg.model import save_model
from emg.rejection import apply_confidence_rejection


CLASSES = ("fist", "rest", "splayed", "wrist_down", "wrist_up")
MINIMUM_ACTIVE_COVERAGE = 0.70


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path("data/collection/compact_A_20261006/collection.json"),
    )
    parser.add_argument(
        "--evaluation-session",
        type=Path,
        default=Path("recordings/S01/compact_A_20261006_4_unfamiliar"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/compact_A_20261006_five_core_shrinkage_v2.joblib"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/five_core_candidate_v2")
    )
    return parser.parse_args()


def new_model():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto",
                    priors=np.full(len(CLASSES), 1.0 / len(CLASSES)),
                ),
            ),
        ]
    )


def selected_indices(dataset, *, session=None, phases=("gesture",)):
    allowed = set(CLASSES)
    return np.asarray(
        [
            index
            for index, window in enumerate(dataset.windows)
            if window.phase in phases
            and dataset.data.labels[index] in allowed
            and (session is None or window.session_id == session)
        ],
        dtype=int,
    )


def sequence_ids(dataset, indices):
    return np.asarray([dataset.windows[index].group_id for index in indices])


def metric_row(threshold, policy, truth, probabilities, groups):
    predicted, rejected = apply_confidence_rejection(
        probabilities, CLASSES, groups, threshold, policy=policy
    )
    metrics = calculate_metrics(truth, predicted, groups, CLASSES, .04)
    accepted = ~rejected
    active = truth != "rest"
    return {
        "threshold": threshold,
        "policy": policy,
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "rest_false_active_fraction": metrics["false_active_window_fraction"],
        "coverage": float(np.mean(accepted)),
        "active_coverage": float(np.mean(accepted[active])),
        "accepted_accuracy": float(np.mean(predicted[accepted] == truth[accepted])) if np.any(accepted) else None,
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classification_and_rest(dataset, indices, predictions):
    gesture = np.asarray([dataset.windows[index].phase == "gesture" for index in indices])
    gesture_indices = indices[gesture]
    groups = sequence_ids(dataset, indices)
    classification = calculate_metrics(
        dataset.data.labels[gesture_indices],
        predictions[gesture],
        groups[gesture],
        CLASSES,
        .04,
    )
    rest_windows = [dataset.windows[index] for index in indices[~gesture]]
    independent = independent_rest_metrics(predictions[~gesture], rest_windows, .04)
    return classification, independent


def trial_majorities(dataset, indices, predictions):
    rows = []
    gesture = np.asarray([dataset.windows[index].phase == "gesture" for index in indices])
    for trial_id in sorted({dataset.windows[index].trial_id for index in indices[gesture]}):
        mask = np.asarray([dataset.windows[index].trial_id == trial_id for index in indices])
        truth = str(dataset.data.labels[indices[mask]][0])
        labels, counts = np.unique(predictions[mask], return_counts=True)
        majority = str(labels[np.argmax(counts)])
        rows.append(
            {
                "trial_id": trial_id,
                "true_label": truth,
                "majority_prediction": majority,
                "majority_correct": majority == truth,
                "correct_window_fraction": float(np.mean(predictions[mask] == truth)),
            }
        )
    return rows


def run(args):
    output = args.output.resolve()
    model_output = args.model_output.resolve()
    if output.exists() or model_output.exists():
        raise ValueError("Use fresh output and model paths; existing artifacts are preserved.")

    _, recordings = read_collection(args.collection)
    dataset, _, fixed_scale = make_dataset(recordings, normalization="fixed_reference")
    sessions = sorted({w.session_id for w in dataset.windows if w.phase == "gesture"})
    all_truth, all_probabilities, all_groups = [], [], []
    fold_rows = []
    for held_out in sessions:
        train = selected_indices(dataset)
        train = np.asarray([i for i in train if dataset.windows[i].session_id != held_out])
        test = selected_indices(
            dataset,
            session=held_out,
            phases=("gesture", "rest_pre", "rest_post"),
        )
        model = new_model().fit(dataset.data.features[train], dataset.data.labels[train])
        result = evaluate_model(model, dataset, test, DEFAULT_CONFIG)
        all_truth.append(dataset.data.labels[result.test_indices])
        all_probabilities.append(result.smoothed_probabilities)
        all_groups.append(sequence_ids(dataset, result.test_indices))
        fold_rows.append(
            {
                "held_out_session": held_out,
                "train_trials": len(set(dataset.data.groups[train])),
                "test_trials": len(set(dataset.data.groups[test])),
                "base_macro_f1": result.metrics["smoothed"]["macro_f1"],
                "base_balanced_accuracy": result.metrics["smoothed"]["balanced_accuracy"],
            }
        )

    truth = np.concatenate(all_truth)
    probabilities = np.vstack(all_probabilities)
    groups = np.concatenate(all_groups)
    thresholds = np.round(np.concatenate(([0.0], np.arange(.50, .951, .01))), 2)
    sweep = [
        metric_row(float(threshold), policy, truth, probabilities, groups)
        for policy in ("hold", "rest")
        for threshold in thresholds
    ]
    eligible = [
        row
        for row in sweep
        if row["policy"] == "hold" and row["active_coverage"] >= MINIMUM_ACTIVE_COVERAGE
    ]
    selected = max(eligible, key=lambda row: (row["macro_f1"], -row["threshold"]))

    deployment = selected_indices(dataset)
    model = new_model().fit(dataset.data.features[deployment], dataset.data.labels[deployment])
    save_model(
        model_output,
        model,
        window_size=25,
        hop_size=5,
        thresholds=0.0,
        sessions=[str(recording.path) for recording in recordings],
        bandpass_hz=(10.0, 55.0),
        sample_rate_hz=125.0,
        spatial_reference="common_average",
        baseline_normalize=False,
    )
    artifact = joblib.load(model_output)
    artifact["compact_profile"] = {
        "version": 1,
        "notch_hz": None,
        "notch_q": 30.0,
        "normalization": "fixed_reference",
        "fixed_reference_scale": fixed_scale.tolist(),
        "filter_reset_guard_samples": 38,
        "smoothing_windows": 7,
    }
    artifact["rejection_profile"] = {
        "version": 1,
        "policy": selected["policy"],
        "minimum_output_confidence": selected["threshold"],
        "selection_protocol": "leave-one-recorded-position-out on training recordings",
        "selection_metric": "smoothed macro F1",
        "selection_constraint": f"hold policy with active coverage >= {MINIMUM_ACTIVE_COVERAGE:.2f}",
        "selection_metrics": selected,
    }
    artifact["training_provenance"] = {
        "experiment": "five_core_shrinkage_candidate_v2",
        "class_set": list(CLASSES),
        "sources": [recording.audit for recording in recordings],
        "deployment_training_trial_ids": sorted(set(dataset.data.groups[deployment])),
        "threshold_selection_folds": sessions,
        "unfamiliar_session_used_for_fitting_or_selection": False,
    }
    joblib.dump(artifact, model_output)

    # This session was already inspected for the seven-class model, so this is a
    # development check rather than a new untouched final test.
    evaluation = read_recording(args.evaluation_session, purpose="held_out_evaluation")
    evaluation_data, _, _ = make_dataset(
        [evaluation],
        normalization="fixed_reference",
        fixed_scale=fixed_scale,
    )
    evaluation_indices = selected_indices(
        evaluation_data, phases=("gesture", "rest_pre", "rest_post")
    )
    evaluation_result = evaluate_model(
        model, evaluation_data, evaluation_indices, DEFAULT_CONFIG
    )
    evaluation_groups = sequence_ids(evaluation_data, evaluation_result.test_indices)
    base_predictions = evaluation_result.smoothed_predictions
    rejected_predictions, rejected_mask = apply_confidence_rejection(
        evaluation_result.smoothed_probabilities,
        CLASSES,
        evaluation_groups,
        selected["threshold"],
        policy=selected["policy"],
    )
    base_classification, base_rest = classification_and_rest(
        evaluation_data, evaluation_result.test_indices, base_predictions
    )
    rejected_classification, rejected_rest = classification_and_rest(
        evaluation_data, evaluation_result.test_indices, rejected_predictions
    )
    trial_scores = trial_majorities(
        evaluation_data, evaluation_result.test_indices, rejected_predictions
    )

    output.mkdir(parents=True)
    write_csv(output / "threshold_sweep.csv", sweep)
    write_csv(output / "position_folds.csv", fold_rows)
    model_hash = hashlib.sha256(model_output.read_bytes()).hexdigest()
    development_check = {
        "warning": "Previously inspected session; development check, not untouched final test.",
        "session": evaluation.audit,
        "base": {"classification": base_classification, "independent_rest": base_rest},
        "with_rejection": {
            "classification": rejected_classification,
            "independent_rest": rejected_rest,
            "rejected_window_fraction": float(np.mean(rejected_mask)),
            "trial_scores": trial_scores,
            "majority_correct_trials": sum(row["majority_correct"] for row in trial_scores),
        },
    }
    (output / "unfamiliar_development_check.json").write_text(
        json.dumps(development_check, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "model": str(model_output),
        "model_sha256": model_hash,
        "classes": list(CLASSES),
        "classifier": "StandardScaler + shrinkage LDA (lsqr, automatic shrinkage, uniform priors)",
        "training_trials": len(set(dataset.data.groups[deployment])),
        "training_windows": len(deployment),
        "selected_rejection": selected,
        "unfamiliar_development_check": {
            "base_macro_f1": base_classification["macro_f1"],
            "rejected_macro_f1": rejected_classification["macro_f1"],
            "base_balanced_accuracy": base_classification["balanced_accuracy"],
            "rejected_balanced_accuracy": rejected_classification["balanced_accuracy"],
            "base_rest_false_active_fraction": base_rest["false_active_window_fraction"],
            "rejected_rest_false_active_fraction": rejected_rest["false_active_window_fraction"],
            "majority_correct_trials": development_check["with_rejection"]["majority_correct_trials"],
            "test_trials": len(trial_scores),
        },
        "limitations": [
            "Same subject and unchanged electrode application only.",
            "Static holds only; no dynamic transition validation.",
            "Unfamiliar-session numbers are not an untouched final test because that session was previously inspected.",
        ],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
