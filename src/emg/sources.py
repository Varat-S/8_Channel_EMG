"""Interchangeable sample sources for hardware-free and live development."""

from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path
from typing import Iterator, Protocol

from .protocol import FrameDecoder
from .types import CHANNEL_COUNT, SAMPLE_RATE_HZ, Sample


class SampleSource(Protocol):
    unit: str

    def read_sample(self) -> Sample:
        """Return the next eight-channel sample or raise EOFError."""


class SyntheticSource:
    """Deterministic gesture-like signals for exercising software, not modelling EMG."""

    unit = "synthetic_count"

    _patterns = {
        "rest": (0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08),
        "fist": (1.00, 0.90, 0.70, 0.40, 0.30, 0.50, 0.80, 0.95),
        "open_hand": (0.30, 0.45, 0.95, 1.00, 0.85, 0.40, 0.25, 0.20),
        "pinch": (0.90, 0.75, 0.25, 0.20, 0.25, 0.35, 0.45, 0.55),
    }

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self._index = 0
        self._gesture = "rest"

    @property
    def supported_gestures(self) -> tuple[str, ...]:
        return tuple(self._patterns)

    def set_gesture(self, gesture: str) -> None:
        if gesture not in self._patterns:
            raise ValueError(f"unsupported synthetic gesture: {gesture}")
        self._gesture = gesture

    def read_sample(self) -> Sample:
        t = self._index / SAMPLE_RATE_HZ
        amplitudes = self._patterns[self._gesture]
        values = tuple(
            amplitude
            * (
                7000.0 * math.sin(2.0 * math.pi * (20.0 + channel * 2.0) * t)
                + 3500.0 * math.sin(2.0 * math.pi * (51.0 - channel * 2.0) * t)
            )
            + self._rng.gauss(0.0, 350.0)
            for channel, amplitude in enumerate(amplitudes)
        )
        sample = Sample(index=self._index, values=values)
        self._index += 1
        return sample


class TextReplaySource:
    """Replay the supplied comma-separated text files using uniform sample timing."""

    unit = "V"

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._file = self._path.open("r", encoding="utf-8", newline="")
        self._rows: Iterator[dict[str, str]] = iter(csv.DictReader(self._file))
        self._index = 0

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> TextReplaySource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read_sample(self) -> Sample:
        try:
            row = next(self._rows)
        except StopIteration as error:
            raise EOFError(f"end of replay file: {self._path}") from error

        # Deliberately ignore Time(s); saved timestamps are known to be batched.
        values = tuple(float(row[f"CH{channel}(V)"]) for channel in range(CHANNEL_COUNT))
        sample = Sample(index=self._index, values=values)
        self._index += 1
        return sample


class SerialFrameSource:
    """Read hypothesized integer channel values from live controller frames."""

    unit = "unscaled_adc_count"

    def __init__(
        self,
        port: str = "COM10",
        baud: int = 921_600,
        timeout: float = 0.5,
    ) -> None:
        import serial

        self._serial_module = serial
        self._port = port
        self._baud = baud
        self._timeout = timeout
        self._connection = self._open_connection()
        self._decoder = FrameDecoder()
        self._pending = []
        self._index = 0
        self._bytes_received = 0
        self.reconnect_count = 0

    @property
    def decoder(self) -> FrameDecoder:
        return self._decoder

    @property
    def bytes_received(self) -> int:
        return self._bytes_received

    def close(self) -> None:
        self._connection.close()

    def reconnect(self) -> None:
        """Close and reopen the port after a receiver/driver stream stall."""

        self._connection.close()
        # The CH340K port can remain open yet stop delivering data. Short close/
        # open cycles did not reset it reliably, while a later fresh process did.
        # Give successive recovery attempts 2, 4, then 6 seconds while closed.
        delay_seconds = min(6.0, 2.0 * (self.reconnect_count + 1))
        time.sleep(delay_seconds)
        self._connection = self._open_connection()
        self._decoder = FrameDecoder()
        self._pending.clear()
        self.reconnect_count += 1

    def _open_connection(self):
        return self._serial_module.Serial(
            port=self._port,
            baudrate=self._baud,
            timeout=self._timeout,
        )

    def __enter__(self) -> SerialFrameSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read_sample(self) -> Sample:
        while not self._pending:
            # Drain the OS buffer when data is already available. Reading one byte
            # when it is empty blocks only until the configured timeout and avoids
            # imposing a fixed 256-byte batching delay on rate measurements.
            bytes_waiting = self._connection.in_waiting
            data = self._connection.read(bytes_waiting if bytes_waiting > 0 else 1)
            if not data:
                raise TimeoutError("no complete controller frame received before timeout")
            self._bytes_received += len(data)
            self._pending.extend(self._decoder.feed(data))

        frame = self._pending.pop(0)
        sample = Sample(
            index=self._index,
            values=tuple(float(value) for value in frame.channel_counts),
            frame_word=frame.metadata_word,
        )
        self._index += 1
        return sample
