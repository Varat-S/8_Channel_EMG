"""Evaluate and adapt the frozen five-core model after electrode reapplication.

The split is registered before any scoring: of six trials per class, three form a
nested calibration pool and three remain the same fixed evaluation set for every
row in the adaptation curve.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from emg.compact_model import make_dataset, read_collection, read_recording
from emg.dataset_v2 import FIVE_CORE_GESTURES
from emg.evaluation.benchmark import DEFAULT_CONFIG, evaluate_model
from emg.evaluation.dataset_v2 import independent_rest_metrics
from emg.evaluation.metrics import calculate_metrics
from emg.model import load_model
from emg.rejection import apply_confidence_rejection


DEFAULT_MODEL = Path("models/compact_A_20261006_five_core_shrinkage_v2.joblib")
DEFAULT_COLLECTION = Path("data/collection/compact_A_20261006/collection.json")
DEFAULT_SESSION = Path("recordings/S01/five_core_reapply_P3")
DEFAULT_OUTPUT = Path("results/five_core_reapplication_P3_v1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print the frozen evaluation design without reading recordings",
    )
    return parser.parse_args()


def new_model(class_count):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto",
                    priors=np.full(class_count, 1.0 / class_count),
                ),
            ),
        ]
    )


def preregistered_trial_split(dataset, seed=23, *, all_trials=None):
    """Return the original seeded split, then remove audited exclusions.

    ``all_trials`` preserves the six-trial pre-registered permutation even when
    the dataset intentionally omits a trial after acquisition. An excluded fixed
    evaluation trial is never silently replaced.
    """
    generator = np.random.default_rng(seed)
    calibration, evaluation = {}, {}
    gesture = np.asarray([window.phase == "gesture" for window in dataset.windows])
    available = {str(value) for value in dataset.data.groups[gesture]}
    for label in FIVE_CORE_GESTURES:
        if all_trials is None:
            ids = sorted(set(dataset.data.groups[gesture & (dataset.data.labels == label)]))
        else:
            ids = sorted(
                trial["trial_id"] for trial in all_trials if trial["gesture"] == label
            )
        if len(ids) != 6:
            raise ValueError(f"{label}: expected six originally recorded trials, found {len(ids)}")
        ordered = [str(value) for value in generator.permutation(ids)]
        calibration[label] = [trial_id for trial_id in ordered[:3] if trial_id in available]
        evaluation[label] = [trial_id for trial_id in ordered[3:] if trial_id in available]
        if len(evaluation[label]) != 3:
            raise ValueError(
                f"{label}: an audited exclusion falls in the fixed evaluation set; "
                "do not replace it after acquisition"
            )
        if not calibration[label]:
            raise ValueError(f"{label}: no eligible calibration trial remains")
    if set(sum(calibration.values(), [])) & set(sum(evaluation.values(), [])):
        raise AssertionError("calibration and fixed evaluation trials overlap")
    return calibration, evaluation


def indices_for_trials(dataset, trial_ids):
    selected = set(trial_ids)
    return np.asarray(
        [
            index
            for index, window in enumerate(dataset.windows)
            if window.phase == "gesture" and window.trial_id in selected
        ],
        dtype=int,
    )


def independent_rest_indices(dataset):
    return np.asarray(
        [
            index
            for index, window in enumerate(dataset.windows)
            if window.phase in ("rest_pre", "rest_post")
        ],
        dtype=int,
    )


def score(model, dataset, test, rest, *, strategy, reps, normalization):
    indices = np.sort(np.concatenate((test, rest)))
    result = evaluate_model(model, dataset, indices, DEFAULT_CONFIG)
    is_gesture = np.asarray(
        [dataset.windows[index].phase == "gesture" for index in result.test_indices]
    )
    groups = np.asarray([dataset.windows[index].group_id for index in result.test_indices])
    classification = calculate_metrics(
        dataset.data.labels[result.test_indices[is_gesture]],
        result.smoothed_predictions[is_gesture],
        groups[is_gesture],
        model.classes_,
        DEFAULT_CONFIG["hop_size"] / DEFAULT_CONFIG["sample_rate_hz"],
    )
    rest_metrics = independent_rest_metrics(
        result.smoothed_predictions[~is_gesture],
        [dataset.windows[index] for index in result.test_indices[~is_gesture]],
        DEFAULT_CONFIG["hop_size"] / DEFAULT_CONFIG["sample_rate_hz"],
    )
    row = {
        "strategy": strategy,
        "calibration_reps_per_class": reps,
        "normalization": normalization,
        "macro_f1": classification["macro_f1"],
        "balanced_accuracy": classification["balanced_accuracy"],
        "accuracy": classification["accuracy"],
        "independent_rest_false_active_fraction": rest_metrics[
            "false_active_window_fraction"
        ],
        "independent_rest_episodes_per_minute": rest_metrics[
            "false_activation_episodes_per_minute"
        ],
        "test_windows": int(np.sum(is_gesture)),
        "independent_rest_windows": int(np.sum(~is_gesture)),
    }
    details = {
        "summary": row,
        "classification": classification,
        "independent_rest": rest_metrics,
    }
    return row, details, result, groups, is_gesture


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan(seed):
    return {
        "design": "five-core electrode-reapplication evaluation",
        "seed": seed,
        "classes": list(FIVE_CORE_GESTURES),
        "required_trials_per_class": 6,
        "calibration_pool_trials_per_class": 3,
        "fixed_evaluation_trials_per_class": 3,
        "calibration_curve_reps_per_class": [1, 2, 3],
        "rows": [
            "strict_zero_shot: frozen model and original fixed normalization",
            "baseline_only: frozen model and new-application baseline normalization",
            "new_only k=1..3: refit from nested target calibration trials",
            "combined k=1..3: refit from original training plus the same target trials",
        ],
        "primary_metric": "smoothed macro F1 on the same 15 untouched trials",
        "threshold_policy": (
            "No threshold tuning on P3. Frozen 0.82 hold policy is reported only as "
            "a supplemental strict-zero-shot live-policy result."
        ),
    }


def run(args):
    if args.plan:
        print(json.dumps(plan(args.seed), indent=2))
        return plan(args.seed)

    output = args.output.resolve()
    if output.exists():
        raise ValueError("use a fresh output directory; evaluation evidence is preserved")
    artifact = load_model(args.model)
    classes = tuple(str(value) for value in artifact["model"].classes_)
    if set(classes) != set(FIVE_CORE_GESTURES):
        raise ValueError("model is not the expected five-core classifier")
    profile = artifact["compact_profile"]

    target_recording = read_recording(
        args.session,
        purpose="held_out_evaluation",
        expected_gestures=FIVE_CORE_GESTURES,
    )
    fixed_target, _, _ = make_dataset(
        [target_recording],
        notch_hz=profile["notch_hz"],
        normalization="fixed_reference",
        fixed_scale=profile["fixed_reference_scale"],
    )
    normalized_target, _, target_scale = make_dataset(
        [target_recording],
        notch_hz=profile["notch_hz"],
        normalization="session_baseline",
    )
    if not np.array_equal(fixed_target.data.labels, normalized_target.data.labels):
        raise AssertionError("normalization variants lost alignment")

    calibration_by_class, evaluation_by_class = preregistered_trial_split(
        fixed_target, args.seed, all_trials=target_recording.metadata["trials"]
    )
    maximum_balanced_reps = min(map(len, calibration_by_class.values()))
    evaluation_ids = sum(evaluation_by_class.values(), [])
    fixed_test = indices_for_trials(fixed_target, evaluation_ids)
    normalized_test = indices_for_trials(normalized_target, evaluation_ids)
    fixed_rest = independent_rest_indices(fixed_target)
    normalized_rest = independent_rest_indices(normalized_target)

    rows, details = [], {}
    strict = score(
        artifact["model"], fixed_target, fixed_test, fixed_rest,
        strategy="strict_zero_shot", reps=0, normalization="original_fixed_reference",
    )
    rows.append(strict[0]); details["strict_zero_shot"] = strict[1]
    baseline = score(
        artifact["model"], normalized_target, normalized_test, normalized_rest,
        strategy="baseline_only", reps=0, normalization="new_application_baseline",
    )
    rows.append(baseline[0]); details["baseline_only"] = baseline[1]

    # Rebuild the exact source feature pool so combined adaptation preserves all
    # deployment trials and does not depend on a historical train/test split.
    _, source_recordings = read_collection(args.collection)
    source_dataset, _, _ = make_dataset(
        source_recordings,
        notch_hz=profile["notch_hz"],
        normalization="fixed_reference",
        fixed_scale=profile["fixed_reference_scale"],
    )
    source_train = np.asarray(
        [
            index
            for index, window in enumerate(source_dataset.windows)
            if window.phase == "gesture" and source_dataset.data.labels[index] in set(classes)
        ],
        dtype=int,
    )
    source_trial_ids = {str(value) for value in source_dataset.data.groups[source_train]}
    expected_source_trial_ids = {
        str(value)
        for value in artifact.get("training_provenance", {}).get(
            "deployment_training_trial_ids", ()
        )
    }
    if source_trial_ids != expected_source_trial_ids:
        raise ValueError(
            "reconstructed source trials differ from the frozen model provenance"
        )

    ordered_calibration = {
        label: calibration_by_class[label] for label in FIVE_CORE_GESTURES
    }
    for reps in range(1, maximum_balanced_reps + 1):
        calibration_ids = sum(
            (ordered_calibration[label][:reps] for label in FIVE_CORE_GESTURES), []
        )
        calibration = indices_for_trials(normalized_target, calibration_ids)
        if set(normalized_target.data.groups[calibration]) & set(
            normalized_target.data.groups[normalized_test]
        ):
            raise AssertionError("target calibration leaked into the fixed test set")
        for strategy in ("new_only", "combined"):
            if strategy == "new_only":
                features = normalized_target.data.features[calibration]
                labels = normalized_target.data.labels[calibration]
            else:
                features = np.vstack(
                    (
                        source_dataset.data.features[source_train],
                        normalized_target.data.features[calibration],
                    )
                )
                labels = np.concatenate(
                    (
                        source_dataset.data.labels[source_train],
                        normalized_target.data.labels[calibration],
                    )
                )
            model = new_model(len(classes)).fit(features, labels)
            scored = score(
                model,
                normalized_target,
                normalized_test,
                normalized_rest,
                strategy=strategy,
                reps=reps,
                normalization="new_application_baseline",
            )
            rows.append(scored[0])
            details[f"{strategy}_reps{reps}"] = {
                **scored[1],
                "calibration_trial_ids": calibration_ids,
                "training_windows": len(features),
            }

    rejection = artifact.get("rejection_profile") or {}
    strict_result, strict_groups, strict_is_gesture = strict[2], strict[3], strict[4]
    live_predictions, rejected = apply_confidence_rejection(
        strict_result.smoothed_probabilities,
        classes,
        strict_groups,
        float(rejection.get("minimum_output_confidence", 0.0)),
        policy=str(rejection.get("policy", "hold")),
    )
    live_classification = calculate_metrics(
        fixed_target.data.labels[strict_result.test_indices[strict_is_gesture]],
        live_predictions[strict_is_gesture],
        strict_groups[strict_is_gesture],
        classes,
        .04,
    )
    live_rest = independent_rest_metrics(
        live_predictions[~strict_is_gesture],
        [fixed_target.windows[index] for index in strict_result.test_indices[~strict_is_gesture]],
        .04,
    )

    output.mkdir(parents=True)
    write_csv(output / "summary.csv", rows)
    actual_plan = plan(args.seed)
    actual_plan["eligible_calibration_trials_per_class"] = {
        label: len(ids) for label, ids in calibration_by_class.items()
    }
    actual_plan["calibration_curve_reps_per_class"] = list(
        range(1, maximum_balanced_reps + 1)
    )
    actual_plan["protocol_deviation"] = (
        "wrist_down repetition 1 was excluded before model scoring following the "
        "participant's timing report and signal-only QC. The original seeded fixed "
        "evaluation trials remain unchanged; the balanced adaptation curve is capped "
        f"at {maximum_balanced_reps} repetitions per class."
        if maximum_balanced_reps < 3 else None
    )
    run_config = {
        **actual_plan,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "collection": str(args.collection.resolve()),
        "collection_sha256": sha256(args.collection),
        "source_training_trials_verified": len(source_trial_ids),
        "target_session": target_recording.audit,
        "calibration_trial_ids_by_class": calibration_by_class,
        "fixed_evaluation_trial_ids_by_class": evaluation_by_class,
        "target_baseline_scale": target_scale.tolist(),
        "original_fixed_reference_scale": profile["fixed_reference_scale"],
        "normalization_note": (
            "Combined adaptation uses original fixed-normalized source features and "
            "new-baseline-normalized target features."
        ),
        "no_model_was_replaced": True,
        "source_sha256": sha256(Path(__file__)),
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    details["strict_zero_shot_live_policy"] = {
        "policy": rejection.get("policy", "hold"),
        "threshold": float(rejection.get("minimum_output_confidence", 0.0)),
        "rejected_window_fraction": float(np.mean(rejected)),
        "classification": live_classification,
        "independent_rest": live_rest,
        "warning": "Supplemental frozen policy; no threshold was selected on P3.",
    }
    (output / "details.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": rows, "output": str(output)}, indent=2), flush=True)
    return {"rows": rows, "details": details, "run_config": run_config}


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, KeyError, OSError, TypeError) as error:
        raise SystemExit(f"Reapplication evaluation error: {error}")
