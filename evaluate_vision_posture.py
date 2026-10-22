"""Evaluate one vision-labelled EMG session across a known posture split."""

from __future__ import annotations

import argparse
import copy
import csv
from collections import deque
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from emg.cnn import SmallEmgCnn
from emg.evaluation.metrics import calculate_metrics
from emg.features import extract_time_domain_features, feature_names
from emg.model import _new_pipeline
from emg.preprocessing import apply_spatial_reference, filter_recording


WINDOW_SIZE = 25
HOP_SIZE = 5
GUARD_SAMPLES = 38
SMOOTHING_WINDOWS = 7
SAMPLE_RATE_HZ = 125.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_session(path: Path) -> dict:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    protocol = json.loads((path / "protocol_posture.json").read_text(encoding="utf-8"))
    if not metadata.get("session_complete") or not metadata.get("training_ready"):
        raise ValueError("session must be complete and training-ready")
    if not protocol.get("authoritative_for_training"):
        raise ValueError("posture protocol is not marked authoritative")
    with (path / "samples.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    values = np.asarray(
        [[float(row[f"CH{channel}"]) for channel in range(8)] for row in rows],
        dtype=np.float64,
    )
    values = apply_spatial_reference(values, "common_average")
    values = filter_recording(
        values, sample_rate_hz=SAMPLE_RATE_HZ, low_hz=10.0, high_hz=55.0
    )
    baseline = np.asarray([row["phase"] == "baseline" for row in rows])
    scale = np.sqrt(np.mean(values[baseline] ** 2, axis=0))
    if not baseline.any() or np.any(scale <= np.finfo(float).eps):
        raise ValueError("valid nonzero baseline is required")
    values /= scale

    trial_metadata = {trial["trial_id"]: trial for trial in metadata["trials"]}
    windows, features, labels, groups, postures, repetitions = [], [], [], [], [], []
    for trial_id, trial in trial_metadata.items():
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row["phase"] == "gesture" and row["trial"] == trial_id
            ],
            dtype=int,
        )
        if len(indices) != trial["sample_count"]:
            raise ValueError(f"sample count mismatch for {trial_id}")
        indices = indices[GUARD_SAMPLES:-GUARD_SAMPLES]
        repetition = int(trial["repetition_index"])
        posture = "vertical" if repetition <= 3 else "horizontal"
        for offset in range(0, len(indices) - WINDOW_SIZE + 1, HOP_SIZE):
            window = values[indices[offset : offset + WINDOW_SIZE]]
            windows.append(window.T.copy())
            features.append(extract_time_domain_features(window))
            labels.append(trial["gesture"])
            groups.append(trial_id)
            postures.append(posture)
            repetitions.append(repetition)
    result = {
        "metadata": metadata,
        "protocol": protocol,
        "baseline_scale": scale,
        "windows": np.stack(windows).astype(np.float32),
        "features": np.stack(features),
        "labels": np.asarray(labels),
        "groups": np.asarray(groups),
        "postures": np.asarray(postures),
        "repetitions": np.asarray(repetitions),
    }
    expected = 42 * ((312 - 2 * GUARD_SAMPLES - WINDOW_SIZE) // HOP_SIZE + 1)
    if len(labels) != expected:
        raise ValueError(f"expected {expected} windows, found {len(labels)}")
    return result


def smooth_probabilities(probabilities: np.ndarray, groups: np.ndarray) -> np.ndarray:
    history: deque[np.ndarray] = deque(maxlen=SMOOTHING_WINDOWS)
    smoothed = []
    previous = None
    for probability, group in zip(probabilities, groups):
        if group != previous:
            history.clear()
        history.append(probability)
        smoothed.append(np.mean(history, axis=0))
        previous = group
    return np.vstack(smoothed)


def score(
    labels: tuple[str, ...], truth: np.ndarray, groups: np.ndarray, probabilities: np.ndarray
) -> dict:
    smoothed = smooth_probabilities(probabilities, groups)
    raw_prediction = np.asarray(labels)[np.argmax(probabilities, axis=1)]
    smooth_prediction = np.asarray(labels)[np.argmax(smoothed, axis=1)]
    return {
        "windows": int(len(truth)),
        "raw": calculate_metrics(truth, raw_prediction, groups, labels, HOP_SIZE / SAMPLE_RATE_HZ),
        "smoothed": calculate_metrics(
            truth, smooth_prediction, groups, labels, HOP_SIZE / SAMPLE_RATE_HZ
        ),
    }


def split_masks(data: dict) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    posture, repetition = data["postures"], data["repetitions"]
    all_indices = np.arange(len(posture))
    result = {
        "vertical_to_horizontal": [
            (all_indices[posture == "vertical"], all_indices[posture == "horizontal"])
        ],
        "horizontal_to_vertical": [
            (all_indices[posture == "horizontal"], all_indices[posture == "vertical"])
        ],
        "mixed_posture_grouped_3fold": [],
    }
    for fold in range(3):
        held_out = np.isin(repetition, (fold + 1, fold + 4))
        result["mixed_posture_grouped_3fold"].append(
            (all_indices[~held_out], all_indices[held_out])
        )
    return result


def lda_experiment(data: dict, folds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    labels = tuple(sorted(set(data["labels"])))
    probabilities = np.zeros((len(data["labels"]), len(labels)), dtype=float)
    test_mask = np.zeros(len(data["labels"]), dtype=bool)
    for train, test in folds:
        if set(data["groups"][train]) & set(data["groups"][test]):
            raise ValueError("trial leakage in LDA split")
        model = _new_pipeline(len(labels)).fit(data["features"][train], data["labels"][train])
        if tuple(model.classes_) != labels:
            raise ValueError("LDA class order mismatch")
        probabilities[test] = model.predict_proba(data["features"][test])
        test_mask[test] = True
    test = np.flatnonzero(test_mask)
    return score(labels, data["labels"][test], data["groups"][test], probabilities[test])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, loss_function, optimizer, device) -> None:
    model.train()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss_function(model(inputs), targets).backward()
        optimizer.step()


def predict_probabilities(model, windows, mean, std, device) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor((windows - mean) / std, dtype=torch.float32)),
        batch_size=128,
        shuffle=False,
    )
    probabilities = []
    with torch.no_grad():
        for (inputs,) in loader:
            probabilities.append(torch.softmax(model(inputs.to(device)), dim=1).cpu().numpy())
    return np.vstack(probabilities)


def validation_indices(data: dict, train: np.ndarray) -> np.ndarray:
    selected_groups = []
    labels = data["labels"]
    for label in sorted(set(labels[train])):
        for posture in sorted(set(data["postures"][train])):
            candidates = sorted(
                set(data["groups"][train][(labels[train] == label) & (data["postures"][train] == posture)])
            )
            if candidates:
                selected_groups.append(candidates[-1])
    return train[np.isin(data["groups"][train], selected_groups)]


def fit_cnn(
    data: dict,
    train: np.ndarray,
    test: np.ndarray,
    *,
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[np.ndarray, int, str]:
    labels = tuple(sorted(set(data["labels"])))
    label_to_index = {label: index for index, label in enumerate(labels)}
    validation = validation_indices(data, train)
    subtrain = train[~np.isin(train, validation)]
    if set(data["groups"][subtrain]) & set(data["groups"][validation]):
        raise ValueError("trial leakage in CNN validation split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    mean = data["windows"][subtrain].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(data["windows"][subtrain].std(axis=(0, 2), keepdims=True), 1e-8)

    def loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        inputs = torch.tensor((data["windows"][indices] - mean) / std, dtype=torch.float32)
        targets = torch.tensor([label_to_index[x] for x in data["labels"][indices]], dtype=torch.long)
        return DataLoader(TensorDataset(inputs, targets), batch_size=64, shuffle=shuffle)

    model = SmallEmgCnn(len(labels)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state, best_f1, best_epoch, stale = copy.deepcopy(model.state_dict()), -1.0, 1, 0
    for epoch in range(1, epochs + 1):
        train_epoch(model, loader(subtrain, True), loss_function, optimizer, device)
        probabilities = predict_probabilities(model, data["windows"][validation], mean, std, device)
        current = score(labels, data["labels"][validation], data["groups"][validation], probabilities)["smoothed"]["macro_f1"]
        if current > best_f1 + 1e-6:
            best_state, best_f1, best_epoch, stale = copy.deepcopy(model.state_dict()), current, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break

    # Refit on every training repetition for the selected epoch count.
    seed_everything(seed)
    mean = data["windows"][train].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(data["windows"][train].std(axis=(0, 2), keepdims=True), 1e-8)
    model = SmallEmgCnn(len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    full_inputs = torch.tensor((data["windows"][train] - mean) / std, dtype=torch.float32)
    full_targets = torch.tensor([label_to_index[x] for x in data["labels"][train]], dtype=torch.long)
    full_loader = DataLoader(TensorDataset(full_inputs, full_targets), batch_size=64, shuffle=True)
    for _ in range(best_epoch):
        train_epoch(model, full_loader, loss_function, optimizer, device)
    return predict_probabilities(model, data["windows"][test], mean, std, device), best_epoch, str(device)


def cnn_experiment(data: dict, folds, *, epochs: int, patience: int, seed: int) -> dict:
    labels = tuple(sorted(set(data["labels"])))
    probabilities = np.zeros((len(data["labels"]), len(labels)), dtype=float)
    test_mask = np.zeros(len(data["labels"]), dtype=bool)
    selected_epochs, devices = [], []
    for fold, (train, test) in enumerate(folds):
        if set(data["groups"][train]) & set(data["groups"][test]):
            raise ValueError("trial leakage in CNN split")
        fold_probabilities, best_epoch, device = fit_cnn(
            data, train, test, epochs=epochs, patience=patience, seed=seed + fold
        )
        probabilities[test] = fold_probabilities
        test_mask[test] = True
        selected_epochs.append(best_epoch)
        devices.append(device)
    test = np.flatnonzero(test_mask)
    result = score(labels, data["labels"][test], data["groups"][test], probabilities[test])
    result["selected_epochs"] = selected_epochs
    result["device"] = sorted(set(devices))
    return result


def save_deployment_models(data: dict, destination: Path, cnn_epochs: int, seed: int) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    labels = tuple(sorted(set(data["labels"])))
    lda = _new_pipeline(len(labels)).fit(data["features"], data["labels"])
    joblib.dump(
        {
            "format_version": 1,
            "model": lda,
            "feature_names": feature_names(),
            "window_size": WINDOW_SIZE,
            "hop_size": HOP_SIZE,
            "guard_samples": GUARD_SAMPLES,
            "bandpass_hz": (10.0, 55.0),
            "spatial_reference": "common_average",
            "baseline_normalize": True,
            "classes": labels,
            "training_session": data["metadata"]["session_id"],
            "posture_protocol": data["protocol"],
        },
        destination / "lda_all_postures.joblib",
    )
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = data["windows"].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(data["windows"].std(axis=(0, 2), keepdims=True), 1e-8)
    targets = {label: index for index, label in enumerate(labels)}
    loader = DataLoader(
        TensorDataset(
            torch.tensor((data["windows"] - mean) / std, dtype=torch.float32),
            torch.tensor([targets[x] for x in data["labels"]], dtype=torch.long),
        ),
        batch_size=64,
        shuffle=True,
    )
    model = SmallEmgCnn(len(labels)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(max(1, cnn_epochs)):
        train_epoch(model, loader, loss_function, optimizer, device)
    torch.save(
        {
            "format_version": 1,
            "state_dict": model.cpu().state_dict(),
            "classes": labels,
            "channel_mean": mean.astype(np.float32),
            "channel_std": std.astype(np.float32),
            "window_size": WINDOW_SIZE,
            "hop_size": HOP_SIZE,
            "guard_samples": GUARD_SAMPLES,
            "bandpass_hz": (10.0, 55.0),
            "spatial_reference": "common_average",
            "baseline_normalize": True,
            "training_session": data["metadata"]["session_id"],
            "posture_protocol": data["protocol"],
            "training_epochs": max(1, cnn_epochs),
        },
        destination / "cnn_all_postures.pt",
    )


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.model_output.exists():
        raise ValueError("output paths already exist; results are never overwritten")
    data = load_session(args.session)
    splits = split_masks(data)
    summary = {
        "session": str(args.session.resolve()),
        "window_count": int(len(data["labels"])),
        "trial_count": int(len(set(data["groups"]))),
        "classes": sorted(set(data["labels"])),
        "posture_ground_truth": data["protocol"],
        "preprocessing": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "bandpass_hz": [10.0, 55.0],
            "spatial_reference": "common_average",
            "baseline_normalization": "per-channel RMS",
            "window_size_samples": WINDOW_SIZE,
            "hop_size_samples": HOP_SIZE,
            "guard_samples_each_edge": GUARD_SAMPLES,
            "smoothing_windows": SMOOTHING_WINDOWS,
        },
        "models": {"lda": {}, "cnn": {}},
    }
    for name, folds in splits.items():
        print(f"LDA {name}", flush=True)
        summary["models"]["lda"][name] = lda_experiment(data, folds)
    cnn_epochs = []
    for name, folds in splits.items():
        print(f"CNN {name}", flush=True)
        result = cnn_experiment(
            data, folds, epochs=args.epochs, patience=args.patience, seed=args.seed
        )
        cnn_epochs.extend(result["selected_epochs"])
        summary["models"]["cnn"][name] = result
    deployment_epochs = max(1, round(float(np.median(cnn_epochs))))
    summary["cnn_deployment_epochs"] = deployment_epochs
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    rows = []
    for model, experiments in summary["models"].items():
        for experiment, result in experiments.items():
            rows.append(
                {
                    "model": model,
                    "experiment": experiment,
                    "test_windows": result["windows"],
                    "raw_accuracy": result["raw"]["accuracy"],
                    "raw_macro_f1": result["raw"]["macro_f1"],
                    "smoothed_accuracy": result["smoothed"]["accuracy"],
                    "smoothed_balanced_accuracy": result["smoothed"]["balanced_accuracy"],
                    "smoothed_macro_f1": result["smoothed"]["macro_f1"],
                }
            )
    with (args.output / "experiments.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_deployment_models(data, args.model_output, deployment_epochs, args.seed)
    print(json.dumps(rows, indent=2), flush=True)
    print(f"Saved report to {args.output}", flush=True)
    print(f"Saved deployment candidates to {args.model_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
