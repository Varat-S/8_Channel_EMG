"""Classification, switching and rest episodes for ordered independent sequences."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)


def calculate_metrics(true, predicted, groups, labels, hop_seconds: float) -> dict:
    true, predicted, groups = map(np.asarray, (true, predicted, groups))
    if true.ndim != 1 or len(true) == 0 or true.shape != predicted.shape or true.shape != groups.shape:
        raise ValueError("true, predicted and groups must be nonempty matching vectors")
    if not np.isfinite(hop_seconds) or hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive and finite")
    if not (set(true) | set(predicted)).issubset(labels):
        raise ValueError("predictions or ground truth contain unknown classes")
    # Inputs must be ordered within each group. Never compare distinct groups.
    switches = intervals = episodes = 0
    for group in dict.fromkeys(groups.tolist()):
        mask = groups == group
        sequence = predicted[mask]
        truth = true[mask]
        switches += int(np.sum(sequence[1:] != sequence[:-1]))
        intervals += max(0, len(sequence) - 1)
        active_rest = (truth == "rest") & (sequence != "rest")
        # A false-active first rest window is a left-censored episode, counted once.
        previous = np.concatenate(([False], active_rest[:-1]))
        episodes += int(np.sum(active_rest & ~previous))
    rest_windows = int(np.sum(true == "rest"))
    false_windows = int(np.sum((true == "rest") & (predicted != "rest")))
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=labels, zero_division=0,
    )
    switch_seconds = intervals * hop_seconds
    rest_seconds = rest_windows * hop_seconds
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0)),
        "labels": list(labels),
        "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist(),
        "per_class": [{"label": str(label), "precision": float(p), "recall": float(r),
                       "f1": float(f), "support": int(n)}
                      for label, p, r, f, n in zip(labels, precision, recall, f1, support)],
        "switch_count": switches,
        "switch_observation_seconds": switch_seconds,
        "switches_per_second": switches / switch_seconds if switch_seconds else None,
        "rest_windows": rest_windows,
        "false_active_windows": false_windows,
        "false_activation_fraction": false_windows / rest_windows if rest_windows else None,
        "false_active_window_fraction": false_windows / rest_windows if rest_windows else None,
        "false_activation_episodes": episodes,
        "rest_observation_seconds": rest_seconds,
        "false_activation_episodes_per_minute": episodes * 60 / rest_seconds if rest_seconds else None,
        "false_activations_per_minute": episodes * 60 / rest_seconds if rest_seconds else None,
    }
