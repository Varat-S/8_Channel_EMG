"""Run the saved feature/LDA pipeline on synthetic, replayed, or live samples."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from emg.features import extract_time_domain_features
from emg.model import load_model
from emg.preprocessing import CausalBandpass, apply_spatial_reference
from emg.sources import SerialFrameSource, SyntheticSource, TextReplaySource
from emg.types import SAMPLE_RATE_HZ


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/lda.joblib"))
    parser.add_argument(
        "--source", choices=("synthetic", "replay", "serial"), default="synthetic"
    )
    parser.add_argument("--gesture", default="fist")
    parser.add_argument("--replay-file", type=Path)
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--baud", type=int, default=921_600)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="stop after this many samples; use 0 for no limit",
    )
    parser.add_argument(
        "--display-every", type=int, default=5, help="print every Nth prediction"
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=5.0,
        help="relaxed baseline used when the model requests normalization",
    )
    args = parser.parse_args()
    if args.source == "replay" and args.replay_file is None:
        parser.error("--replay-file is required for replay")
    if args.max_samples < 0 or args.display_every <= 0:
        parser.error("max-samples cannot be negative and display-every must be positive")
    return args


def make_source(args: argparse.Namespace):
    if args.source == "synthetic":
        source = SyntheticSource(seed=19)
        source.set_gesture(args.gesture)
        return nullcontext(source)
    if args.source == "replay":
        return TextReplaySource(args.replay_file)
    return SerialFrameSource(port=args.port, baud=args.baud)


def main() -> int:
    args = parse_args()
    artifact = load_model(args.model)
    model = artifact["model"]
    window_size = int(artifact["window_size"])
    hop_size = int(artifact["hop_size"])
    thresholds = artifact["thresholds"]
    bandpass_hz = artifact.get("bandpass_hz")
    spatial_reference = artifact.get("spatial_reference", "none")
    baseline_normalize = bool(artifact.get("baseline_normalize", False))
    bandpass = (
        CausalBandpass(
            sample_rate_hz=float(artifact["sample_rate_hz"]),
            low_hz=float(bandpass_hz[0]),
            high_hz=float(bandpass_hz[1]),
        )
        if bandpass_hz is not None
        else None
    )
    buffer: deque[tuple[float, ...]] = deque(maxlen=window_size)
    samples_read = 0
    predictions_made = 0

    print(
        f"Loaded {args.model}; window={window_size}, hop={hop_size}, "
        f"source={args.source}"
    )
    try:
        with make_source(args) as source:
            baseline_scale = np.ones(8, dtype=np.float64)
            if baseline_normalize:
                baseline_count = round(
                    args.baseline_seconds * float(artifact["sample_rate_hz"])
                )
                print(
                    f"Remain relaxed and still for {args.baseline_seconds:g} seconds...",
                    flush=True,
                )
                baseline_values = []
                for _ in range(baseline_count):
                    sample = source.read_sample()
                    referenced = apply_spatial_reference(
                        sample.values, spatial_reference
                    )
                    processed = (
                        bandpass.update(referenced) if bandpass is not None else referenced
                    )
                    baseline_values.append(processed)
                baseline_scale = np.sqrt(
                    np.mean(np.asarray(baseline_values) ** 2, axis=0)
                )
                baseline_scale = np.maximum(baseline_scale, 1e-8)
                print("Baseline complete; live predictions starting.", flush=True)

            while args.max_samples == 0 or samples_read < args.max_samples:
                try:
                    sample = source.read_sample()
                except EOFError:
                    print("Replay complete.")
                    break
                referenced = apply_spatial_reference(sample.values, spatial_reference)
                processed = (
                    bandpass.update(referenced)
                    if bandpass is not None
                    else referenced
                )
                processed = np.asarray(processed) / baseline_scale
                buffer.append(tuple(processed))
                samples_read += 1

                if len(buffer) < window_size:
                    continue
                if (samples_read - window_size) % hop_size != 0:
                    continue

                features = extract_time_domain_features(
                    np.asarray(buffer, dtype=np.float64), thresholds
                )
                probabilities = model.predict_proba(features.reshape(1, -1))[0]
                best = int(np.argmax(probabilities))
                predictions_made += 1
                if predictions_made % args.display_every == 0:
                    print(
                        f"t={sample.index / SAMPLE_RATE_HZ:8.3f}s  "
                        f"prediction={model.classes_[best]:<12} "
                        f"score={probabilities[best]:.3f}"
                    )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except TimeoutError as error:
        print(f"Input timeout: {error}", file=sys.stderr)
        return 1

    print(f"Read {samples_read} samples and made {predictions_made} predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
