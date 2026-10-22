"""Compare compact EMG class sets and classifiers without changing deployment.

All splits keep complete gesture trials together. The already-inspected unfamiliar
recording is deliberately excluded: this script is for model selection, not a new
final-test claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from emg.cnn import SmallEmgCnn
from emg.compact_model import compact_split, make_dataset, read_collection
from emg.evaluation.benchmark import DEFAULT_CONFIG, evaluate_model
from emg.evaluation.dataset import BenchmarkDataset


CLASS_SETS = {
    "five_core": ("fist", "rest", "splayed", "wrist_down", "wrist_up"),
    "seven_full": ("fist", "pinch", "point", "rest", "splayed", "wrist_down", "wrist_up"),
}
SEEDS = tuple(range(7, 12))


class FixedProbabilities:
    """Adapter allowing the shared temporal evaluator to score CNN probabilities."""

    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, features):
        if len(features) != len(self.probabilities):
            raise ValueError("precomputed probabilities do not match requested rows")
        return self.probabilities


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path("data/collection/compact_A_20261006/collection.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/compact_model_benchmark_v1")
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def _classical_models(class_count, seed):
    priors = np.full(class_count, 1.0 / class_count)
    return {
        "lda": Pipeline(
            [("scale", StandardScaler()), ("model", LinearDiscriminantAnalysis(priors=priors))]
        ),
        "shrinkage_lda": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LinearDiscriminantAnalysis(
                        solver="lsqr", shrinkage="auto", priors=priors
                    ),
                ),
            ]
        ),
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0, class_weight="balanced", max_iter=2000, random_state=seed
                    ),
                ),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=3.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _loader(windows, targets, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.tensor(windows, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _train_epochs(model, loader, weights, epochs):
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()


def _cnn_probabilities(
    signal_windows,
    labels,
    groups,
    train_indices,
    test_indices,
    classes,
    seed,
    epochs,
    patience,
    batch_size,
):
    """Select epoch count on inner whole trials, then refit all outer training rows."""
    _seed(seed)
    class_to_index = {label: index for index, label in enumerate(classes)}
    outer_labels = labels[train_indices]
    outer_groups = groups[train_indices]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    inner_train_local, validation_local = next(
        splitter.split(signal_windows[train_indices], outer_labels, outer_groups)
    )
    inner_train = train_indices[inner_train_local]
    validation = train_indices[validation_local]
    for name, indices in (("inner training", inner_train), ("validation", validation)):
        if set(labels[indices]) != set(classes):
            raise ValueError(f"CNN {name} fold is missing classes")

    mean = signal_windows[inner_train].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(signal_windows[inner_train].std(axis=(0, 2), keepdims=True), 1e-8)
    normalized = (signal_windows - mean) / std
    train_targets = np.asarray([class_to_index[x] for x in labels[inner_train]])
    validation_targets = np.asarray([class_to_index[x] for x in labels[validation]])
    counts = np.bincount(train_targets, minlength=len(classes))
    weights = len(train_targets) / (len(classes) * counts)
    train_loader = _loader(
        normalized[inner_train], train_targets, batch_size, True, seed
    )
    validation_loader = _loader(
        normalized[validation], validation_targets, batch_size, False, seed
    )
    model = SmallEmgCnn(len(classes))
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_f1, best_epoch, stale = -1.0, 1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
        probabilities = _predict_cnn(model, validation_loader)
        predicted = np.argmax(probabilities, axis=1)
        score = f1_score(
            validation_targets,
            predicted,
            labels=np.arange(len(classes)),
            average="macro",
            zero_division=0,
        )
        if score > best_f1 + 1e-6:
            best_f1, best_epoch, stale = float(score), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break

    # Refit from scratch on every outer-training trial for the selected epoch count.
    _seed(seed + 1000)
    mean = signal_windows[train_indices].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(signal_windows[train_indices].std(axis=(0, 2), keepdims=True), 1e-8)
    normalized = (signal_windows - mean) / std
    full_targets = np.asarray([class_to_index[x] for x in labels[train_indices]])
    counts = np.bincount(full_targets, minlength=len(classes))
    weights = len(full_targets) / (len(classes) * counts)
    full_loader = _loader(
        normalized[train_indices], full_targets, batch_size, True, seed + 1000
    )
    model = SmallEmgCnn(len(classes))
    _train_epochs(model, full_loader, weights, best_epoch)
    test_targets = np.asarray([class_to_index[x] for x in labels[test_indices]])
    test_loader = _loader(
        normalized[test_indices], test_targets, batch_size, False, seed
    )
    return _predict_cnn(model, test_loader), best_epoch, best_f1


def _predict_cnn(model, loader):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for inputs, _ in loader:
            probabilities.append(torch.softmax(model(inputs), dim=1).numpy())
    return np.vstack(probabilities)


def _select(indices, labels, class_labels):
    allowed = set(class_labels)
    return np.asarray([i for i in indices if labels[i] in allowed], dtype=int)


def _row(protocol, class_set, model_name, fold, result, train, test, extra=None):
    row = {
        "protocol": protocol,
        "class_set": class_set,
        "model": model_name,
        "fold": str(fold),
        "train_trials": len(set(dataset_global.data.groups[train])),
        "test_trials": len(set(dataset_global.data.groups[test])),
        "train_windows": len(train),
        "test_windows": len(test),
        "raw_macro_f1": result.metrics["raw"]["macro_f1"],
        "raw_balanced_accuracy": result.metrics["raw"]["balanced_accuracy"],
        "smoothed_macro_f1": result.metrics["smoothed"]["macro_f1"],
        "smoothed_balanced_accuracy": result.metrics["smoothed"]["balanced_accuracy"],
        "cnn_best_epoch": "",
        "cnn_inner_validation_f1": "",
    }
    if extra:
        row.update(extra)
    return row


def _evaluate_one(model_name, model, train, test, dataset):
    model.fit(dataset.data.features[train], dataset.data.labels[train])
    return evaluate_model(model, dataset, test, DEFAULT_CONFIG)


def _aggregate(rows, protocol):
    groups = {}
    for row in rows:
        if row["protocol"] != protocol:
            continue
        groups.setdefault((row["class_set"], row["model"]), []).append(row)
    output = []
    for (class_set, model), selected in sorted(groups.items()):
        f1 = np.asarray([r["smoothed_macro_f1"] for r in selected])
        accuracy = np.asarray([r["smoothed_balanced_accuracy"] for r in selected])
        output.append(
            {
                "class_set": class_set,
                "model": model,
                "folds": len(selected),
                "smoothed_macro_f1_mean": float(f1.mean()),
                "smoothed_macro_f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
                "smoothed_balanced_accuracy_mean": float(accuracy.mean()),
                "smoothed_balanced_accuracy_std": float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
            }
        )
    return output


def _write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows):
    lines = [
        "| Class set | Model | Folds | Smoothed macro F1 | Balanced accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class_set']} | {row['model']} | {row['folds']} | "
            f"{row['smoothed_macro_f1_mean']:.3f} +/- {row['smoothed_macro_f1_std']:.3f} | "
            f"{row['smoothed_balanced_accuracy_mean']:.3f} +/- {row['smoothed_balanced_accuracy_std']:.3f} |"
        )
    return "\n".join(lines)


def run(args):
    global dataset_global
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory; benchmark reports are preserved.")
    collection, recordings = read_collection(args.collection)
    dataset, conditions, scale, signal_windows = make_dataset(
        recordings,
        normalization="fixed_reference",
        include_signal_windows=True,
    )
    dataset_global = dataset
    labels = dataset.data.labels
    gesture = np.asarray([w.phase == "gesture" for w in dataset.windows])
    all_gesture = np.flatnonzero(gesture)
    rows, details = [], []

    for seed in SEEDS:
        full_train, full_test = compact_split(dataset, seed)
        for set_name, classes in CLASS_SETS.items():
            train = _select(full_train, labels, classes)
            test = _select(full_test, labels, classes)
            for model_name, model in _classical_models(len(classes), seed).items():
                result = _evaluate_one(model_name, model, train, test, dataset)
                rows.append(_row("whole_trial", set_name, model_name, seed, result, train, test))
                details.append({"protocol":"whole_trial","class_set":set_name,"model":model_name,"fold":seed,"metrics":result.metrics})
            probabilities, best_epoch, validation_f1 = _cnn_probabilities(
                signal_windows, labels, dataset.data.groups, train, test, classes,
                seed, args.epochs, args.patience, args.batch_size,
            )
            result = evaluate_model(FixedProbabilities(classes, probabilities), dataset, test, DEFAULT_CONFIG)
            rows.append(_row("whole_trial", set_name, "cnn", seed, result, train, test,
                             {"cnn_best_epoch":best_epoch,"cnn_inner_validation_f1":validation_f1}))
            details.append({"protocol":"whole_trial","class_set":set_name,"model":"cnn","fold":seed,
                            "best_epoch":best_epoch,"inner_validation_f1":validation_f1,"metrics":result.metrics})
            print(f"whole_trial seed={seed} class_set={set_name} complete", flush=True)

    sessions = sorted({w.session_id for w in dataset.windows if w.phase == "gesture"})
    for fold_number, held_out in enumerate(sessions):
        outer_train = np.asarray([i for i in all_gesture if dataset.windows[i].session_id != held_out])
        outer_test = np.asarray([i for i in all_gesture if dataset.windows[i].session_id == held_out])
        for set_name, classes in CLASS_SETS.items():
            train = _select(outer_train, labels, classes)
            test = _select(outer_test, labels, classes)
            for model_name, model in _classical_models(len(classes), 100 + fold_number).items():
                result = _evaluate_one(model_name, model, train, test, dataset)
                rows.append(_row("leave_one_position_out", set_name, model_name, held_out, result, train, test))
                details.append({"protocol":"leave_one_position_out","class_set":set_name,"model":model_name,
                                "fold":held_out,"metrics":result.metrics})
            probabilities, best_epoch, validation_f1 = _cnn_probabilities(
                signal_windows, labels, dataset.data.groups, train, test, classes,
                100 + fold_number, args.epochs, args.patience, args.batch_size,
            )
            result = evaluate_model(FixedProbabilities(classes, probabilities), dataset, test, DEFAULT_CONFIG)
            rows.append(_row("leave_one_position_out", set_name, "cnn", held_out, result, train, test,
                             {"cnn_best_epoch":best_epoch,"cnn_inner_validation_f1":validation_f1}))
            details.append({"protocol":"leave_one_position_out","class_set":set_name,"model":"cnn","fold":held_out,
                            "best_epoch":best_epoch,"inner_validation_f1":validation_f1,"metrics":result.metrics})
            print(f"leave_one_position_out held_out={held_out} class_set={set_name} complete", flush=True)

    output.mkdir(parents=True)
    _write_csv(output / "fold_metrics.csv", rows)
    (output / "fold_details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    whole = _aggregate(rows, "whole_trial")
    position = _aggregate(rows, "leave_one_position_out")
    summary = {
        "purpose": "model selection only; unfamiliar evaluation session excluded",
        "class_sets": CLASS_SETS,
        "whole_trial": whole,
        "leave_one_position_out": position,
        "recording_sources": [recording.audit for recording in recordings],
        "fixed_reference_scale": scale.tolist(),
        "seeds": SEEDS,
        "cnn": {"max_epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = f"""# Compact EMG model benchmark

This is a model-selection benchmark, not a new final test. Every split keeps complete
gesture trials together. The previously inspected unfamiliar-position recording is
excluded from fitting, epoch selection, and scoring.

## Repeated whole-trial validation

{_markdown_table(whole)}

## Leave-one-recorded-position-out validation

{_markdown_table(position)}

The five-class comparison excludes pinch and point. CNN epoch counts are selected on
an inner whole-trial validation fold, after which the network is reinitialized and
refit on all outer-training trials. All models use the selected fixed-reference
normalization, causal 10-55 Hz filtering, common-average reference, 25-sample windows,
five-sample hops, and seven-window probability smoothing.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return summary


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments)
