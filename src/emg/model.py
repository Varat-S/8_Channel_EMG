"""LDA training, evaluation, and artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import FeatureDataset
from .features import feature_names


@dataclass(frozen=True, slots=True)
class Evaluation:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    labels: tuple[str, ...]
    confusion_matrix: NDArray[np.int64]
    train_windows: int
    test_windows: int


def train_and_evaluate(
    dataset: FeatureDataset,
    *,
    random_state: int = 7,
) -> tuple[Pipeline, Evaluation]:
    labels = tuple(sorted(set(dataset.labels)))
    if len(labels) < 2:
        raise ValueError("LDA requires at least two gesture classes")

    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    try:
        train_indices, test_indices = next(
            splitter.split(dataset.features, dataset.labels, dataset.groups)
        )
    except ValueError as error:
        raise ValueError(
            "not enough independent labelled segments for a grouped train/test split"
        ) from error

    train_labels = set(dataset.labels[train_indices])
    test_labels = set(dataset.labels[test_indices])
    if train_labels != set(labels) or test_labels != set(labels):
        raise ValueError(
            "grouped split did not contain every class on both sides; record more trials"
        )

    evaluation_model = _new_pipeline(len(labels))
    evaluation_model.fit(dataset.features[train_indices], dataset.labels[train_indices])
    predictions = evaluation_model.predict(dataset.features[test_indices])
    evaluation = Evaluation(
        accuracy=float(accuracy_score(dataset.labels[test_indices], predictions)),
        balanced_accuracy=float(
            balanced_accuracy_score(dataset.labels[test_indices], predictions)
        ),
        macro_f1=float(
            f1_score(dataset.labels[test_indices], predictions, average="macro")
        ),
        labels=labels,
        confusion_matrix=confusion_matrix(
            dataset.labels[test_indices], predictions, labels=labels
        ),
        train_windows=len(train_indices),
        test_windows=len(test_indices),
    )

    # The deployable model uses all supplied data. Its training-set performance is
    # never reported as evaluation accuracy.
    final_model = _new_pipeline(len(labels))
    final_model.fit(dataset.features, dataset.labels)
    return final_model, evaluation


def train_and_evaluate_holdout(
    training: FeatureDataset,
    testing: FeatureDataset,
) -> tuple[Pipeline, Evaluation]:
    """Fit only on training sessions and evaluate on untouched test sessions."""

    labels = tuple(sorted(set(training.labels)))
    missing_from_training = set(testing.labels) - set(labels)
    missing_from_testing = set(labels) - set(testing.labels)
    if missing_from_training or missing_from_testing:
        raise ValueError(
            "train/test class sets differ: "
            f"only_test={sorted(missing_from_training)}, "
            f"only_train={sorted(missing_from_testing)}"
        )

    model = _new_pipeline(len(labels))
    model.fit(training.features, training.labels)
    predictions = model.predict(testing.features)
    evaluation = Evaluation(
        accuracy=float(accuracy_score(testing.labels, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(testing.labels, predictions)),
        macro_f1=float(f1_score(testing.labels, predictions, average="macro")),
        labels=labels,
        confusion_matrix=confusion_matrix(testing.labels, predictions, labels=labels),
        train_windows=len(training.labels),
        test_windows=len(testing.labels),
    )
    return model, evaluation


def save_model(
    path: str | Path,
    model: Pipeline,
    *,
    window_size: int,
    hop_size: int,
    thresholds: float | list[float],
    sessions: list[str],
    bandpass_hz: tuple[float, float] | None,
    sample_rate_hz: float,
    spatial_reference: str,
    baseline_normalize: bool,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact: dict[str, Any] = {
        "format_version": 1,
        "model": model,
        "window_size": window_size,
        "hop_size": hop_size,
        "thresholds": thresholds,
        "feature_names": feature_names(),
        "sessions": sessions,
        "bandpass_hz": bandpass_hz,
        "sample_rate_hz": sample_rate_hz,
        "spatial_reference": spatial_reference,
        "baseline_normalize": baseline_normalize,
    }
    joblib.dump(artifact, destination)


def load_model(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if artifact.get("format_version") != 1:
        raise ValueError("unsupported model artifact format")
    return artifact


def _new_pipeline(class_count: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lda",
                LinearDiscriminantAnalysis(
                    priors=np.full(class_count, 1.0 / class_count)
                ),
            ),
        ]
    )
