"""Confidence rejection policies for ordered EMG prediction sequences."""
from __future__ import annotations

import numpy as np


def apply_confidence_rejection(
    probabilities,
    classes,
    sequence_ids,
    threshold: float,
    *,
    policy: str,
):
    """Return output labels and a mask identifying rejected predictions.

    ``hold`` keeps the most recent accepted output within a continuous sequence,
    starting each new sequence at rest. ``rest`` maps every rejection to rest.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)
    sequence_ids = np.asarray(sequence_ids)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("probabilities must have one column per class")
    if len(probabilities) == 0 or len(sequence_ids) != len(probabilities):
        raise ValueError("probabilities and sequence IDs must be nonempty and aligned")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if policy not in {"hold", "rest"}:
        raise ValueError("policy must be 'hold' or 'rest'")
    if "rest" not in set(classes.tolist()):
        raise ValueError("rejection requires a rest class")

    confidence = np.max(probabilities, axis=1)
    predicted = classes[np.argmax(probabilities, axis=1)]
    rejected = confidence < threshold
    if policy == "rest":
        return np.where(rejected, "rest", predicted), rejected

    output = []
    previous_sequence = None
    accepted = "rest"
    for candidate, reject, sequence in zip(predicted, rejected, sequence_ids):
        if previous_sequence is None or sequence != previous_sequence:
            accepted = "rest"
        if not reject:
            accepted = str(candidate)
        output.append(accepted)
        previous_sequence = sequence
    return np.asarray(output), rejected
