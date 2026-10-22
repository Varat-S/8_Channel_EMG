"""Train LDA on complete sessions and test on separate untouched sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emg.dataset import load_feature_dataset
from emg.model import train_and_evaluate_holdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", required=True, type=Path)
    parser.add_argument("--test", nargs="+", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--hop-size", type=int, default=5)
    parser.add_argument("--low-hz", type=float, default=10.0)
    parser.add_argument("--high-hz", type=float, default=55.0)
    parser.add_argument("--guard-ms", type=float, default=300.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--spatial-reference", choices=("none", "common_average"), default="none"
    )
    parser.add_argument(
        "--baseline-normalize",
        action="store_true",
        help="divide each filtered channel by its session rest-baseline RMS",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = json.loads(
        (args.train[0] / "metadata.json").read_text(encoding="utf-8")
    )
    sample_rate_hz = float(metadata["sample_rate_hz"])
    common = {
        "window_size": args.window_size,
        "hop_size": args.hop_size,
        "thresholds": args.threshold,
        "bandpass_hz": (args.low_hz, args.high_hz),
        "guard_samples": round(args.guard_ms * sample_rate_hz / 1000.0),
        "spatial_reference": args.spatial_reference,
        "baseline_normalize": args.baseline_normalize,
    }
    training = load_feature_dataset(args.train, **common)
    testing = load_feature_dataset(args.test, **common)
    _, evaluation = train_and_evaluate_holdout(training, testing)

    print("Training sessions:", ", ".join(str(path) for path in args.train))
    print("Test sessions:", ", ".join(str(path) for path in args.test))
    print(
        f"Windows: {evaluation.train_windows} train, "
        f"{evaluation.test_windows} untouched test"
    )
    print(f"Holdout accuracy: {evaluation.accuracy:.3f}")
    print(f"Balanced accuracy: {evaluation.balanced_accuracy:.3f}")
    print(f"Macro-F1: {evaluation.macro_f1:.3f}")
    print("Labels:", ", ".join(evaluation.labels))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(evaluation.confusion_matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
