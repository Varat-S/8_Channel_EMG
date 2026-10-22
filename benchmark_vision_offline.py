"""Held-out seven-gesture robustness, preprocessing ablation, and trial intervals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from emg.cnn import SmallEmgCnn
from emg.features import extract_time_domain_features
from emg.model import _new_pipeline
from emg.preprocessing import apply_spatial_reference, filter_recording
from evaluate_vision_posture import (GUARD_SAMPLES, HOP_SIZE, SAMPLE_RATE_HZ, WINDOW_SIZE,
                                     load_session, predict_probabilities, score, seed_everything,
                                     smooth_probabilities, split_masks, train_epoch)


def trial_bootstrap(truth, predicted, groups, labels, *, postures=None, draws=4000, seed=7):
    """Resample whole trials within gesture and, when given, posture strata."""
    rng = np.random.default_rng(seed)
    trial_ids = np.unique(groups)
    if postures is None:
        postures = np.repeat("all", len(groups))
    strata = {(label, posture): [group for group in trial_ids
                                if truth[groups == group][0] == label
                                and postures[groups == group][0] == posture]
              for label in labels for posture in np.unique(postures)}
    strata = {key: value for key, value in strata.items() if value}
    scores = np.empty(draws)
    trial_indices = {group: np.flatnonzero(groups == group) for group in trial_ids}
    for draw in range(draws):
        selected = np.concatenate([trial_indices[group]
                                   for trial_set in strata.values()
                                   for group in rng.choice(trial_set, size=len(trial_set), replace=True)])
        scores[draw] = f1_score(truth[selected], predicted[selected], labels=labels,
                                average="macro", zero_division=0)
    return {"point": float(f1_score(truth, predicted, labels=labels, average="macro")),
            "lower_95": float(np.quantile(scores, .025)),
            "upper_95": float(np.quantile(scores, .975)),
            "unit": "whole trial; stratified by gesture and posture when available",
            "draws": draws, "seed": seed}


def corruption_conditions(windows, seed):
    rng = np.random.default_rng(seed)
    yield "clean", windows
    for channel in range(8):
        altered = windows.copy()
        altered[:, channel] = 0
        yield f"drop_ch{channel + 1}", altered
        altered = windows.copy()
        altered[:, channel] *= .5
        yield f"half_gain_ch{channel + 1}", altered
    for gain in (.5, 1.5):
        yield f"all_gain_{gain}", windows * gain
    rms = np.sqrt(np.mean(windows ** 2, axis=(0, 2), keepdims=True))
    for snr in (20, 10):
        yield f"noise_{snr}db", windows + rng.normal(size=windows.shape).astype(np.float32) * rms * 10 ** (-snr / 20)
    for offset in (1, 2):
        altered = windows.copy()
        altered[:, :, offset:] = windows[:, :, :-offset]
        altered[:, :, :offset] = windows[:, :, :1]
        yield f"timing_lag_{offset}_samples", altered


def train_fold(data, train, epochs, seed):
    labels = tuple(sorted(set(data["labels"])))
    seed_everything(seed)
    mean = data["windows"][train].mean(axis=(0, 2), keepdims=True)
    std = np.maximum(data["windows"][train].std(axis=(0, 2), keepdims=True), 1e-8)
    model = SmallEmgCnn(len(labels))
    loader = DataLoader(TensorDataset(
        torch.tensor((data["windows"][train] - mean) / std, dtype=torch.float32),
        torch.tensor([labels.index(label) for label in data["labels"][train]], dtype=torch.long)),
        batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        train_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"))
    return model, mean, std


def cnn_benchmark(data, epochs, seed):
    folds = split_masks(data)["mixed_posture_grouped_3fold"]
    labels = tuple(sorted(set(data["labels"])))
    conditions = [name for name, _ in corruption_conditions(data["windows"][:1], seed)]
    probabilities = {name: np.zeros((len(data["labels"]), len(labels)), dtype=np.float32)
                     for name in conditions}
    for fold, (train, test) in enumerate(folds):
        if set(data["groups"][train]) & set(data["groups"][test]):
            raise ValueError("trial leakage")
        print(f"CNN fold {fold + 1}/3, {epochs[fold]} epochs", flush=True)
        model, mean, std = train_fold(data, train, epochs[fold], seed + fold)
        for name, altered in corruption_conditions(data["windows"][test], seed + fold):
            probabilities[name][test] = predict_probabilities(model, altered, mean, std, torch.device("cpu"))
    rows = []
    for name, predicted_probs in probabilities.items():
        metrics = score(labels, data["labels"], data["groups"], predicted_probs)
        rows.append({"condition": name, "raw_macro_f1": metrics["raw"]["macro_f1"],
                     "smoothed_macro_f1": metrics["smoothed"]["macro_f1"],
                     "smoothed_accuracy": metrics["smoothed"]["accuracy"]})
    smoothed = smooth_probabilities(probabilities["clean"], data["groups"])
    predicted = np.asarray(labels)[np.argmax(smoothed, axis=1)]
    interval = trial_bootstrap(data["labels"], predicted, data["groups"], labels,
                               postures=data["postures"], seed=seed)
    return rows, interval


def alternate_windows(session, data, *, spatial_reference=True, bandpass=True, baseline_normalize=True):
    with (session / "samples.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    raw = np.array([[float(row[f"CH{i}"]) for i in range(8)] for row in rows])
    values = apply_spatial_reference(raw, "common_average" if spatial_reference else "none")
    if bandpass:
        values = filter_recording(values, sample_rate_hz=SAMPLE_RATE_HZ, low_hz=10, high_hz=55)
    if baseline_normalize:
        baseline = np.asarray([row["phase"] == "baseline" for row in rows])
        scale = np.sqrt(np.mean(values[baseline] ** 2, axis=0))
        if np.any(scale < 1e-8):
            raise ValueError("invalid baseline scale")
        values /= scale
    windows = []
    for trial in data["metadata"]["trials"]:
        indices = np.array([i for i, row in enumerate(rows)
                            if row["phase"] == "gesture" and row["trial"] == trial["trial_id"]])
        indices = indices[GUARD_SAMPLES:-GUARD_SAMPLES]
        for offset in range(0, len(indices) - WINDOW_SIZE + 1, HOP_SIZE):
            windows.append(values[indices[offset:offset + WINDOW_SIZE]])
    return np.stack(windows)


def lda_ablations(session, data):
    variants = {
        "reference": {},
        "no_common_average": {"spatial_reference": False},
        "no_bandpass": {"bandpass": False},
        "no_baseline_normalization": {"baseline_normalize": False},
    }
    labels = tuple(sorted(set(data["labels"])))
    folds = split_masks(data)["mixed_posture_grouped_3fold"]
    rows = []
    for name, settings in variants.items():
        windows = alternate_windows(session, data, **settings)
        features = np.stack([extract_time_domain_features(window) for window in windows])
        probabilities = np.zeros((len(features), len(labels)))
        for train, test in folds:
            model = _new_pipeline(len(labels)).fit(features[train], data["labels"][train])
            probabilities[test] = model.predict_proba(features[test])
        metrics = score(labels, data["labels"], data["groups"], probabilities)
        rows.append({"variant": name, "raw_macro_f1": metrics["raw"]["macro_f1"],
                     "smoothed_macro_f1": metrics["smoothed"]["macro_f1"]})
    return rows


def cnn_ablations(session, data, epochs, seed, reference_score):
    labels = tuple(sorted(set(data["labels"])))
    folds = split_masks(data)["mixed_posture_grouped_3fold"]
    rows = [{"variant": "reference", "raw_macro_f1": reference_score["raw_macro_f1"],
             "smoothed_macro_f1": reference_score["smoothed_macro_f1"]}]
    variants = {
        "no_common_average": {"spatial_reference": False},
        "no_bandpass": {"bandpass": False},
        "no_baseline_normalization": {"baseline_normalize": False},
    }
    for name, settings in variants.items():
        windows = alternate_windows(session, data, **settings).transpose(0, 2, 1).astype(np.float32)
        variant_data = {**data, "windows": windows}
        probabilities = np.zeros((len(windows), len(labels)), dtype=np.float32)
        for fold, (train, test) in enumerate(folds):
            print(f"CNN {name}: fold {fold + 1}/3", flush=True)
            model, mean, std = train_fold(variant_data, train, epochs[fold], seed + fold)
            probabilities[test] = predict_probabilities(model, windows[test], mean, std,
                                                         torch.device("cpu"))
        metrics = score(labels, data["labels"], data["groups"], probabilities)
        rows.append({"variant": name, "raw_macro_f1": metrics["raw"]["macro_f1"],
                     "smoothed_macro_f1": metrics["smoothed"]["macro_f1"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=Path("recordings/S01/vision_20261016_192456"))
    parser.add_argument("--summary", type=Path, default=Path("results/vision_posture_20261016_192456/summary.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = load_session(args.session)
    original = json.loads(args.summary.read_text(encoding="utf-8"))
    epochs = original["models"]["cnn"]["mixed_posture_grouped_3fold"]["selected_epochs"]
    if len(epochs) != 3:
        raise ValueError("expected three grouped fold epoch counts")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    robustness, interval = cnn_benchmark(data, epochs, args.seed)
    cnn_preprocessing = cnn_ablations(args.session, data, epochs, args.seed, robustness[0])
    lda_preprocessing = lda_ablations(args.session, data)
    result = {"scope": "within-session grouped held-out trials; artificial corruptions, no new application",
              "session": str(args.session), "fold_selected_epochs": epochs,
              "robustness_after_preprocessing": robustness,
              "preprocessing_ablations_cnn": cnn_preprocessing,
              "preprocessing_ablations_lda": lda_preprocessing,
              "mixed_posture_cnn_smoothed_macro_f1_trial_interval": interval}
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("robustness.csv", robustness),
                       ("preprocessing_ablations_cnn.csv", cnn_preprocessing),
                       ("preprocessing_ablations_lda.csv", lda_preprocessing)):
        with (args.output / name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"clean": robustness[0], "trial_interval": interval,
                      "cnn_ablations": cnn_preprocessing, "lda_ablations": lda_preprocessing}, indent=2))


if __name__ == "__main__":
    main()
