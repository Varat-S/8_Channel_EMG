"""Train a small 1D CNN with grouped validation and session-held-out testing."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from emg.cnn import SmallEmgCnn
from emg.dataset import WindowDataset, load_window_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", required=True, type=Path)
    parser.add_argument("--test", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/emg_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _seed_everything(args.seed)
    training = load_window_dataset(args.train)
    testing = load_window_dataset(args.test)
    classes = tuple(sorted(set(training.labels)))
    if set(testing.labels) != set(classes):
        raise ValueError("training and test sessions contain different class sets")
    label_to_index = {label: index for index, label in enumerate(classes)}

    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.seed)
    train_indices, validation_indices = next(
        splitter.split(training.windows, training.labels, training.groups)
    )
    _require_all_classes(training.labels[train_indices], classes, "training fold")
    _require_all_classes(training.labels[validation_indices], classes, "validation fold")

    channel_mean = training.windows[train_indices].mean(axis=(0, 2), keepdims=True)
    channel_std = training.windows[train_indices].std(axis=(0, 2), keepdims=True)
    channel_std = np.maximum(channel_std, 1e-8)
    normalized_training = (training.windows - channel_mean) / channel_std
    normalized_testing = (testing.windows - channel_mean) / channel_std

    train_loader = _loader(
        normalized_training[train_indices],
        training.labels[train_indices],
        label_to_index,
        args.batch_size,
        shuffle=True,
    )
    validation_loader = _loader(
        normalized_training[validation_indices],
        training.labels[validation_indices],
        label_to_index,
        args.batch_size,
        shuffle=False,
    )
    test_loader = _loader(
        normalized_testing,
        testing.labels,
        label_to_index,
        args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallEmgCnn(len(classes)).to(device)
    counts = np.bincount(
        [label_to_index[label] for label in training.labels[train_indices]],
        minlength=len(classes),
    )
    class_weights = len(train_indices) / (len(classes) * counts)
    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_state = copy.deepcopy(model.state_dict())
    best_validation_f1 = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, loss_function, optimizer, device)
        validation_true, validation_pred = _predict(model, validation_loader, device)
        validation_f1 = f1_score(validation_true, validation_pred, average="macro")
        print(
            f"epoch={epoch:03d} loss={train_loss:.4f} "
            f"validation_macro_f1={validation_f1:.3f}"
        )
        if validation_f1 > best_validation_f1 + 1e-6:
            best_validation_f1 = validation_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    model.load_state_dict(best_state)
    test_true, test_pred = _predict(model, test_loader, device)
    print(f"Device: {device}")
    print(f"Best validation macro-F1: {best_validation_f1:.3f}")
    print(f"Untouched test accuracy: {accuracy_score(test_true, test_pred):.3f}")
    print(
        "Untouched test balanced accuracy: "
        f"{balanced_accuracy_score(test_true, test_pred):.3f}"
    )
    print(f"Untouched test macro-F1: {f1_score(test_true, test_pred, average='macro'):.3f}")
    print("Labels:", ", ".join(classes))
    print(confusion_matrix(test_true, test_pred, labels=np.arange(len(classes))))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "state_dict": model.cpu().state_dict(),
            "classes": classes,
            "channel_mean": channel_mean.astype(np.float32),
            "channel_std": channel_std.astype(np.float32),
            "window_size": 25,
            "hop_size": 5,
            "sample_rate_hz": 125,
            "bandpass_hz": (10.0, 55.0),
            "spatial_reference": "common_average",
            "baseline_normalize": True,
            "training_sessions": [str(path) for path in args.train],
            "test_sessions": [str(path) for path in args.test],
        },
        args.output,
    )
    print(f"Saved CNN artifact to {args.output}")
    return 0


def _loader(
    windows: np.ndarray,
    labels: np.ndarray,
    label_to_index: dict[str, int],
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader:
    inputs = torch.tensor(windows, dtype=torch.float32)
    targets = torch.tensor(
        [label_to_index[label] for label in labels], dtype=torch.long
    )
    return DataLoader(
        TensorDataset(inputs, targets), batch_size=batch_size, shuffle=shuffle
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(inputs)
    return total_loss / len(loader.dataset)


def _predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            true.extend(targets.numpy().tolist())
            predicted.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
    return np.asarray(true), np.asarray(predicted)


def _require_all_classes(
    labels: np.ndarray, classes: tuple[str, ...], description: str
) -> None:
    missing = set(classes) - set(labels)
    if missing:
        raise ValueError(f"{description} is missing classes: {sorted(missing)}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())

