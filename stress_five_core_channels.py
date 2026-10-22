"""Approximate channel-placement sensitivity of the frozen five-core candidate.

These deterministic raw-channel transformations are engineering stress tests, not
measurements of physical electrode reapplication.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from emg.compact_model import make_dataset, read_recording
from emg.evaluation.benchmark import DEFAULT_CONFIG, evaluate_model
from emg.evaluation.dataset_v2 import independent_rest_metrics
from emg.evaluation.metrics import calculate_metrics
from emg.model import load_model
from emg.rejection import apply_confidence_rejection


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/compact_A_20261006_five_core_shrinkage_v2.joblib"),
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=Path("recordings/S01/compact_A_20261006_4_unfamiliar"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/five_core_channel_shift_v1")
    )
    return parser.parse_args()


def transformations(values):
    yield "clean", "clean", None, values.copy()
    for shift in range(1, 8):
        yield f"cyclic_shift_{shift}", "cyclic_shift", shift, np.roll(values, shift, axis=1)
    for channel in range(8):
        dropped = values.copy()
        dropped[:, channel] = 0.0
        yield f"drop_ch{channel + 1}", "dropout", channel + 1, dropped
    for gain in (0.5, 1.5):
        for channel in range(8):
            changed = values.copy()
            changed[:, channel] *= gain
            yield f"gain_{gain:g}_ch{channel + 1}", "single_channel_gain", channel + 1, changed
    for gain in (0.5, 0.75, 1.25, 1.5):
        yield f"global_gain_{gain:g}", "global_gain", gain, values * gain


def score_condition(recording, values, artifact):
    profile = artifact["compact_profile"]
    changed = replace(recording, values=np.asarray(values, dtype=float))
    dataset, _, _ = make_dataset(
        [changed],
        notch_hz=profile["notch_hz"],
        normalization=profile["normalization"],
        fixed_scale=profile["fixed_reference_scale"],
    )
    classes = tuple(str(label) for label in artifact["model"].classes_)
    allowed = set(classes)
    indices = np.asarray(
        [
            i
            for i, window in enumerate(dataset.windows)
            if window.phase in ("gesture", "rest_pre", "rest_post")
            and dataset.data.labels[i] in allowed
        ]
    )
    result = evaluate_model(artifact["model"], dataset, indices, DEFAULT_CONFIG)
    groups = np.asarray([dataset.windows[i].group_id for i in result.test_indices])
    gesture = np.asarray([dataset.windows[i].phase == "gesture" for i in result.test_indices])
    gesture_indices = result.test_indices[gesture]

    def metrics(predicted):
        classification = calculate_metrics(
            dataset.data.labels[gesture_indices], predicted[gesture], groups[gesture], classes, .04
        )
        rest = independent_rest_metrics(
            predicted[~gesture],
            [dataset.windows[i] for i in result.test_indices[~gesture]],
            .04,
        )
        return classification, rest

    base, base_rest = metrics(result.smoothed_predictions)
    rejection = artifact.get("rejection_profile") or {}
    rejected, mask = apply_confidence_rejection(
        result.smoothed_probabilities,
        classes,
        groups,
        float(rejection.get("minimum_output_confidence", 0.0)),
        policy=str(rejection.get("policy", "hold")),
    )
    gated, gated_rest = metrics(rejected)
    return {
        "base_macro_f1": base["macro_f1"],
        "base_balanced_accuracy": base["balanced_accuracy"],
        "base_rest_false_active_fraction": base_rest["false_active_window_fraction"],
        "gated_macro_f1": gated["macro_f1"],
        "gated_balanced_accuracy": gated["balanced_accuracy"],
        "gated_rest_false_active_fraction": gated_rest["false_active_window_fraction"],
        "rejected_window_fraction": float(np.mean(mask)),
    }


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory; stress reports are preserved.")
    artifact = load_model(args.model)
    recording = read_recording(args.session, purpose="held_out_evaluation")
    rows = []
    for name, family, parameter, values in transformations(recording.values):
        row = {"condition": name, "family": family, "parameter": parameter}
        row.update(score_condition(recording, values, artifact))
        rows.append(row)
        print(f"{name}: gated macro F1={row['gated_macro_f1']:.3f}", flush=True)

    clean = rows[0]
    for row in rows:
        row["gated_macro_f1_delta_vs_clean"] = row["gated_macro_f1"] - clean["gated_macro_f1"]
    output.mkdir(parents=True)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    worst = sorted(rows[1:], key=lambda row: row["gated_macro_f1"])[:5]
    summary = {
        "warning": "Synthetic channel transformations are not physical reapplication measurements.",
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "session": recording.audit,
        "clean": clean,
        "worst_five": worst,
        "condition_count": len(rows),
        "transformations": {
            "cyclic_shift": "Roll raw CH1..CH8 identities; plausible only if channel order follows a forearm ring.",
            "dropout": "Set one raw channel to zero before CAR and filtering.",
            "single_channel_gain": "Multiply one raw channel by 0.5 or 1.5 before CAR and filtering.",
            "global_gain": "Multiply all raw channels before fixed-reference normalization.",
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    run(parse_args())
