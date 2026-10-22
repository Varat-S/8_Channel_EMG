"""Measure provisional decoded-frame behaviour without attaching electrodes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from serial import SerialException

from emg.sources import SerialFrameSource
from emg.types import CHANNEL_COUNT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--baud", type=int, default=921_600)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    minimums = [float("inf")] * CHANNEL_COUNT
    maximums = [float("-inf")] * CHANNEL_COUNT
    frame_words: set[int] = set()
    frame_count = 0
    started = time.monotonic()

    print(
        "Using the provisional 38-byte/big-endian interpretation; values are "
        "unscaled counts, not confirmed volts."
    )
    try:
        with SerialFrameSource(port=args.port, baud=args.baud) as source:
            while time.monotonic() - started < args.seconds:
                sample = source.read_sample()
                frame_count += 1
                if sample.frame_word is not None:
                    frame_words.add(sample.frame_word)
                for channel, value in enumerate(sample.values):
                    minimums[channel] = min(minimums[channel], value)
                    maximums[channel] = max(maximums[channel], value)

            elapsed = time.monotonic() - started
            print(f"Frames: {frame_count} in {elapsed:.3f} s")
            print(f"Observed frame rate: {frame_count / elapsed:.2f} frames/s")
            print(
                f"Raw serial rate: {source.bytes_received / elapsed:.2f} bytes/s "
                f"({source.bytes_received} bytes)"
            )
            print(
                "Unknown two-byte frame words:",
                ", ".join(f"0x{value:04X}" for value in sorted(frame_words)),
            )
            print("Channel ranges (unscaled signed counts):")
            for channel, (minimum, maximum) in enumerate(zip(minimums, maximums)):
                print(f"  CH{channel}: {minimum:12.0f} .. {maximum:12.0f}")
            print(f"Discarded bytes while synchronizing: {source.decoder.discarded_bytes}")
            print(f"Bad frame candidates: {source.decoder.bad_candidates}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    except (SerialException, OSError, TimeoutError) as error:
        print(f"Frame probe error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
