"""Inspect raw bytes from an EMG controller without assuming a protocol."""

from __future__ import annotations

import argparse
import sys
import time

import serial
from serial import SerialException


DEFAULT_PORT = "COM10"
DEFAULT_BAUD = 921_600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print raw serial chunks as repr() and hexadecimal bytes."
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help="serial port (default: COM10)")
    parser.add_argument(
        "--baud", type=int, default=DEFAULT_BAUD, help="baud rate (default: 921600)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="maximum bytes printed per read (default: 256)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.5,
        help="read timeout in seconds (default: 0.5)",
    )
    args = parser.parse_args()
    if args.baud <= 0 or args.chunk_size <= 0 or args.timeout < 0:
        parser.error("baud and chunk-size must be positive; timeout cannot be negative")
    return args


def probe(port: str, baud: int, chunk_size: int, timeout: float) -> None:
    total_bytes = 0
    started = time.monotonic()
    last_idle_message = started

    # The context manager guarantees that the port closes on exit or Ctrl+C.
    with serial.Serial(port=port, baudrate=baud, timeout=timeout) as connection:
        print(
            f"Opened {connection.port} at {connection.baudrate} baud "
            f"(8N1, timeout={connection.timeout}s)."
        )
        print("Reading only; no initialization commands will be sent. Press Ctrl+C to stop.")

        while True:
            data = connection.read(chunk_size)
            now = time.monotonic()

            if not data:
                if now - last_idle_message >= 5.0:
                    print(f"[{now - started:9.3f}s] no bytes received yet")
                    last_idle_message = now
                continue

            total_bytes += len(data)
            last_idle_message = now
            print(f"\n[{now - started:9.3f}s] chunk={len(data)} total={total_bytes}")
            print(f"repr: {data!r}")
            print(f"hex : {data.hex(' ')}")


def main() -> int:
    args = parse_args()
    try:
        probe(args.port, args.baud, args.chunk_size, args.timeout)
    except KeyboardInterrupt:
        print("\nStopped by user; serial port closed.")
        return 0
    except (SerialException, OSError) as error:
        print(f"Serial error: {error}", file=sys.stderr)
        print(
            "Check that the port name is correct and that the proprietary EMG "
            "software (and any other serial monitor) is closed.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
