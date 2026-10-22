"""Train pooled P3/P4 CNNs with trial-grouped folds and electrode perturbations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from emg.cnn import SmallEmgCnn
from evaluate_vision_posture import load_session, score


VARIANTS = ("plain", "gain", "gain_dropout")
HELD_OUT_REPETITIONS = ((1, 4), (2, 5), (3, 6))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pooled(paths: list[Path]) -> dict:
    if len(paths) != 2:
        raise ValueError("exactly two source applications are required")
    sessions = [load_session(path) for path in paths]
    labels = tuple(sorted(set(sessions[0]["labels"])))
    if len(labels) != 7 or any(tuple(sorted(set(item["labels"]))) != labels for item in sessions):
        raise ValueError("both sessions must have the same seven gestures")
    data = {key: np.concatenate([item[key] for item in sessions])
            for key in ("windows", "labels", "postures", "repetitions")}
    data["sessions"] = np.concatenate([
        np.full(len(item["labels"]), item["metadata"]["session_id"])
        for item in sessions])
    data["groups"] = np.concatenate([
        np.asarray([f"{item['metadata']['session_id']}:{group}" for group in item["groups"]])
        for item in sessions])
    data["labels_order"] = labels
    root = Path(__file__).resolve().parent
    data["session_paths"] = [str(path.resolve().relative_to(root)).replace("\\", "/")
                             for path in paths]
    data["session_ids"] = [item["metadata"]["session_id"] for item in sessions]
    data["metadata_sha256"] = [sha256(path / "metadata.json") for path in paths]
    for session_id in data["session_ids"]:
        mask = data["sessions"] == session_id
        if len(set(data["groups"][mask])) != 42:
            raise ValueError(f"expected 42 complete trials in {session_id}")
    return data


def fold_masks(data: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    result = []
    for repetitions in HELD_OUT_REPETITIONS:
        test = np.isin(data["repetitions"], repetitions)
        train = ~test
        if set(data["groups"][train]) & set(data["groups"][test]):
            raise ValueError("trial leakage")
        result.append((train, test))
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(data: dict, train: np.ndarray, variant: str, epochs: int, seed: int):
    if variant not in VARIANTS:
        raise ValueError(variant)
    seed_everything(seed)
    windows = data["windows"][train]
    labels = data["labels"][train]
    groups = data["groups"][train]
    mean = windows.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.maximum(windows.std(axis=(0, 2), keepdims=True), 1e-8).astype(np.float32)
    classes = data["labels_order"]
    targets = torch.tensor([classes.index(label) for label in labels], dtype=torch.long)
    group_names = sorted(set(groups))
    group_ids = torch.tensor([group_names.index(group) for group in groups], dtype=torch.long)
    loader = DataLoader(TensorDataset(torch.from_numpy(windows), targets, group_ids),
                        batch_size=64, shuffle=True)
    model = SmallEmgCnn(len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    mean_tensor, std_tensor = torch.from_numpy(mean), torch.from_numpy(std)
    log_channel = math.log(1.5)
    log_global = math.log(1.25)
    for _ in range(epochs):
        model.train()
        factors = torch.ones((len(group_names), 8, 1))
        if variant != "plain":
            channel = torch.empty_like(factors).uniform_(-log_channel, log_channel).exp_()
            global_gain = torch.empty((len(group_names), 1, 1)).uniform_(-log_global, log_global).exp_()
            factors *= channel * global_gain
        if variant == "gain_dropout":
            selected = torch.rand(len(group_names)) < 0.15
            channel_ids = torch.randint(0, 8, (len(group_names),))
            factors[torch.arange(len(group_names))[selected], channel_ids[selected], 0] = 0.0
        for inputs, target, group in loader:
            if variant != "plain":
                inputs = inputs * factors[group]
            inputs = (inputs - mean_tensor) / std_tensor
            optimizer.zero_grad()
            loss = loss_function(model(inputs), target)
            loss.backward()
            optimizer.step()
    return model.eval(), mean, std


def predict(model, mean: np.ndarray, std: np.ndarray, windows: np.ndarray) -> np.ndarray:
    probabilities = []
    with torch.no_grad():
        for batch in np.array_split(windows, max(1, (len(windows) + 127) // 128)):
            inputs = torch.tensor((batch - mean) / std, dtype=torch.float32)
            probabilities.append(torch.softmax(model(inputs), dim=1).numpy())
    return np.vstack(probabilities)


def evaluate_variant(data: dict, variant: str, epochs: int, seed: int) -> dict:
    probabilities = np.zeros((len(data["labels"]), len(data["labels_order"])), dtype=np.float32)
    held_out = np.zeros(len(data["labels"]), dtype=bool)
    for fold, (train, test) in enumerate(fold_masks(data)):
        model, mean, std = train_model(data, train, variant, epochs, seed + fold)
        probabilities[test] = predict(model, mean, std, data["windows"][test])
        held_out |= test
    if not held_out.all():
        raise ValueError("folds did not score every window")
    report = {"overall": score(data["labels_order"], data["labels"], data["groups"], probabilities),
              "per_session": {}}
    for session_id in data["session_ids"]:
        mask = data["sessions"] == session_id
        report["per_session"][session_id] = score(
            data["labels_order"], data["labels"][mask], data["groups"][mask], probabilities[mask])
    return report


def save_artifact(data: dict, variant: str, epochs: int, seed: int, destination: Path) -> None:
    model, mean, std = train_model(data, np.ones(len(data["labels"]), dtype=bool), variant, epochs, seed)
    torch.save({
        "format_version": 1, "state_dict": model.state_dict(),
        "classes": data["labels_order"], "channel_mean": mean, "channel_std": std,
        "window_size": 25, "hop_size": 5, "guard_samples": 38,
        "bandpass_hz": (10.0, 55.0), "spatial_reference": "common_average",
        "baseline_normalize": True, "baseline_strategy": "per_training_session_RMS",
        "training_sessions": data["session_ids"], "training_metadata_sha256": data["metadata_sha256"],
        "training_epochs": epochs, "training_variant": variant, "seed": seed,
    }, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs=2, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=31)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists() or args.models.exists():
        raise FileExistsError("results and models are never overwritten")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    data = load_pooled(args.sessions)
    report = {
        "scope": "exploratory within-known-applications grouped whole-trial folds; P5 required for independent validation",
        "sessions": data["session_ids"], "session_paths": data["session_paths"],
        "metadata_sha256": data["metadata_sha256"],
        "fold_test_repetitions": HELD_OUT_REPETITIONS,
        "normalization": "each session divided by its own recorded rest baseline before pooling",
        "training_epochs": args.epochs, "seed": args.seed,
        "augmentation": {
            "gain": "per-trial channel factor log-uniform [1/1.5, 1.5] and global factor [1/1.25, 1.25], resampled each epoch",
            "gain_dropout": "gain plus one post-preprocessing channel zeroed in 15% of trials each epoch",
        },
        "variants": {},
    }
    for variant in VARIANTS:
        print(f"Evaluating {variant}", flush=True)
        report["variants"][variant] = evaluate_variant(data, variant, args.epochs, args.seed)
    args.output.mkdir(parents=True)
    args.models.mkdir(parents=True)
    for variant in VARIANTS:
        print(f"Training full {variant}", flush=True)
        save_artifact(data, variant, args.epochs, args.seed,
                      args.models / f"cnn_{variant}.pt")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("variant", "session", "smoothed_accuracy", "smoothed_macro_f1"))
        writer.writeheader()
        for variant, results in report["variants"].items():
            for session, metrics in (("pooled", results["overall"]), *results["per_session"].items()):
                writer.writerow({"variant": variant, "session": session,
                                 "smoothed_accuracy": metrics["smoothed"]["accuracy"],
                                 "smoothed_macro_f1": metrics["smoothed"]["macro_f1"]})
    print((args.output / "summary.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
