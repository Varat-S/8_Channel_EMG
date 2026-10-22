"""Train and group-evaluate the 32-feature LDA baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emg.dataset import load_feature_dataset
from emg.model import save_model, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/lda.joblib"))
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--hop-size", type=int, default=5)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="provisional ZC/SSC change threshold in the recording's unit",
    )
    parser.add_argument("--low-hz", type=float, default=10.0)
    parser.add_argument("--high-hz", type=float, default=55.0)
    parser.add_argument(
        "--no-filter", action="store_true", help="disable causal band-pass filtering"
    )
    parser.add_argument(
        "--guard-ms",
        type=float,
        default=300.0,
        help="discard this much from both ends of every labelled segment",
    )
    parser.add_argument(
        "--spatial-reference", choices=("none", "common_average"), default="none"
    )
    parser.add_argument(
        "--baseline-normalize",
        action="store_true",
        help="divide each channel by its session rest-baseline RMS",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = json.loads(
        (args.sessions[0] / "metadata.json").read_text(encoding="utf-8")
    )
    sample_rate_hz = float(metadata["sample_rate_hz"])
    bandpass_hz = None if args.no_filter else (args.low_hz, args.high_hz)
    guard_samples = round(args.guard_ms * sample_rate_hz / 1000.0)
    dataset = load_feature_dataset(
        args.sessions,
        window_size=args.window_size,
        hop_size=args.hop_size,
        thresholds=args.threshold,
        bandpass_hz=bandpass_hz,
        guard_samples=guard_samples,
        spatial_reference=args.spatial_reference,
        baseline_normalize=args.baseline_normalize,
    )
    model, evaluation = train_and_evaluate(dataset)
    save_model(
        args.output,
        model,
        window_size=args.window_size,
        hop_size=args.hop_size,
        thresholds=args.threshold,
        sessions=[str(session) for session in args.sessions],
        bandpass_hz=bandpass_hz,
        sample_rate_hz=sample_rate_hz,
        spatial_reference=args.spatial_reference,
        baseline_normalize=args.baseline_normalize,
    )

    print(f"Windows: {len(dataset.labels)}")
    print(f"Grouped split: {evaluation.train_windows} train, {evaluation.test_windows} test")
    print(f"Accuracy: {evaluation.accuracy:.3f}")
    print(f"Balanced accuracy: {evaluation.balanced_accuracy:.3f}")
    print(f"Macro-F1: {evaluation.macro_f1:.3f}")
    print("Labels:", ", ".join(evaluation.labels))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(evaluation.confusion_matrix)
    print(f"Deployable model trained on all windows and saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
