"""Exploratory posture contrastive and 200 ms future-signal CNN experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from benchmark_vision_offline import trial_bootstrap
from evaluate_vision_posture import (load_session, score, seed_everything,
                                     smooth_probabilities, split_masks)
from emg.cnn import SmallEmgCnn


class AuxiliaryCnn(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.backbone = SmallEmgCnn(class_count)
        self.future = nn.Linear(32, 8)

    def forward(self, x):
        representation = self.backbone.features(x).flatten(1)
        logits = self.backbone.classifier(representation)
        return logits, representation, self.future(representation)


def posture_contrastive_loss(representation, label, posture, temperature=.2):
    """InfoNCE positives share gesture and differ in forearm orientation."""
    n = len(label)
    z = F.normalize(representation, dim=1)
    similarities = z @ z.T / temperature
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    positive = (label[:, None] == label[None, :]) & (posture[:, None] != posture[None, :])
    positive &= ~eye
    valid = positive.any(dim=1)
    if not valid.any():
        return representation.sum() * 0
    logits = similarities.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -((log_prob.masked_fill(~positive, 0).sum(dim=1) / positive.sum(dim=1).clamp(min=1))[valid]).mean()


def future_targets(data, horizon=5):
    """Target the RMS of a nonoverlapping window 5 hops (200 ms) ahead."""
    windows, groups = data["windows"], data["groups"]
    targets = np.zeros((len(windows), 8), dtype=np.float32)
    valid = np.zeros(len(windows), dtype=bool)
    for i in range(len(windows) - horizon):
        if groups[i] == groups[i + horizon]:
            targets[i] = np.log1p(np.sqrt(np.mean(windows[i + horizon] ** 2, axis=1)))
            valid[i] = True
    return targets, valid


def train_and_predict(data, train, test, *, variant, epochs, seed, auxiliary_weight):
    seed_everything(seed)
    labels = tuple(sorted(set(data["labels"])))
    mean = data["windows"][train].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(data["windows"][train].std(axis=(0, 2), keepdims=True), 1e-8)
    targets, valid_future = future_targets(data)
    future_train = train[valid_future[train]]
    future_mean = targets[future_train].mean(axis=0)
    future_std = np.maximum(targets[future_train].std(axis=0), 1e-6)
    normalized = (targets[train] - future_mean) / future_std
    loader = DataLoader(TensorDataset(
        torch.tensor((data["windows"][train] - mean) / std, dtype=torch.float32),
        torch.tensor([labels.index(x) for x in data["labels"][train]], dtype=torch.long),
        torch.tensor(data["postures"][train] == "horizontal", dtype=torch.long),
        torch.tensor(normalized, dtype=torch.float32),
        torch.tensor(valid_future[train], dtype=torch.bool)), batch_size=64, shuffle=True)
    model = AuxiliaryCnn(len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for windows, labels_batch, postures, next_rms, valid in loader:
            optimizer.zero_grad()
            logits, representation, forecast = model(windows)
            loss = F.cross_entropy(logits, labels_batch)
            if variant == "cross_posture_contrastive":
                loss = loss + auxiliary_weight * posture_contrastive_loss(representation, labels_batch, postures)
            elif variant == "future_rms_200ms":
                if valid.any():
                    loss = loss + auxiliary_weight * F.mse_loss(forecast[valid], next_rms[valid])
            else:
                raise ValueError(variant)
            loss.backward()
            optimizer.step()
    model.eval()
    test_windows = torch.tensor((data["windows"][test] - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        return np.vstack([torch.softmax(model(batch)[0], dim=1).numpy()
                          for batch in test_windows.split(128)])


def run(data, epochs, seed, weight):
    labels = tuple(sorted(set(data["labels"])))
    folds = split_masks(data)["mixed_posture_grouped_3fold"]
    rows = []
    for variant in ("cross_posture_contrastive", "future_rms_200ms"):
        probabilities = np.zeros((len(data["labels"]), len(labels)), dtype=np.float32)
        for fold, (train, test) in enumerate(folds):
            if set(data["groups"][train]) & set(data["groups"][test]):
                raise ValueError("trial leakage")
            print(f"{variant}: fold {fold + 1}/3", flush=True)
            probabilities[test] = train_and_predict(data, train, test, variant=variant,
                                                     epochs=epochs[fold], seed=seed + fold,
                                                     auxiliary_weight=weight)
        metrics = score(labels, data["labels"], data["groups"], probabilities)
        predicted = np.asarray(labels)[np.argmax(smooth_probabilities(probabilities, data["groups"]), axis=1)]
        interval = trial_bootstrap(data["labels"], predicted, data["groups"], labels,
                                   postures=data["postures"], seed=seed)
        rows.append({"variant": variant, "raw_macro_f1": metrics["raw"]["macro_f1"],
                     "smoothed_macro_f1": metrics["smoothed"]["macro_f1"],
                     "trial_ci_lower_95": interval["lower_95"],
                     "trial_ci_upper_95": interval["upper_95"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=Path("recordings/S01/vision_20261016_192456"))
    parser.add_argument("--summary", type=Path, default=Path("results/vision_posture_20261016_192456/summary.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--auxiliary-weight", type=float, default=.1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.auxiliary_weight < 0:
        raise ValueError("auxiliary weight must be nonnegative")
    data = load_session(args.session)
    original = json.loads(args.summary.read_text(encoding="utf-8"))
    epochs = original["models"]["cnn"]["mixed_posture_grouped_3fold"]["selected_epochs"]
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rows = run(data, epochs, args.seed, args.auxiliary_weight)
    result = {"scope": "exploratory within-session grouped folds; auxiliary loss fixed without tuning",
              "horizon_samples": 25, "horizon_seconds": .2,
              "auxiliary_weight": args.auxiliary_weight, "selected_epochs": epochs,
              "baseline_smoothed_macro_f1": original["models"]["cnn"]["mixed_posture_grouped_3fold"]["smoothed"]["macro_f1"],
              "experiments": rows}
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (args.output / "experiments.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
