"""Incremental decoder for the controller frames observed on 2026-09-13.

Confirmed from the current capture:

* frames are 38 bytes apart;
* they start with 55 66 and end with 66 55;
* there are 34 bytes between those markers.

Current hypothesis (kept isolated here): the first two payload bytes are an unknown
metadata word and the remaining 32 bytes are eight big-endian signed integers. The apparent
sign extension is consistent with 24-bit ADC samples, but neither representation
nor voltage scaling is considered confirmed yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import CHANNEL_COUNT


HEADER = b"\x55\x66"
TRAILER = b"\x66\x55"
FRAME_SIZE = 38
METADATA_SIZE = 2
CHANNEL_BYTES = 4


@dataclass(frozen=True, slots=True)
class ControllerFrame:
    metadata_word: int
    channel_counts: tuple[int, ...]
    raw: bytes


class FrameDecoder:
    """Recover complete frames from arbitrarily split or noisy serial chunks."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.discarded_bytes = 0
        self.bad_candidates = 0

    def feed(self, data: bytes) -> list[ControllerFrame]:
        self._buffer.extend(data)
        frames: list[ControllerFrame] = []

        while True:
            start = self._buffer.find(HEADER)
            if start < 0:
                # Retain a final 0x55 because it could be the start of a header
                # split across two reads.
                keep = 1 if self._buffer.endswith(HEADER[:1]) else 0
                discard = len(self._buffer) - keep
                self.discarded_bytes += discard
                if discard:
                    del self._buffer[:discard]
                break

            if start:
                self.discarded_bytes += start
                del self._buffer[:start]

            if len(self._buffer) < FRAME_SIZE:
                break

            candidate = bytes(self._buffer[:FRAME_SIZE])
            if not candidate.endswith(TRAILER):
                # This header was probably payload/noise. Advance one byte and
                # search again instead of losing a possible following frame.
                self.bad_candidates += 1
                self.discarded_bytes += 1
                del self._buffer[0]
                continue

            frames.append(decode_frame(candidate))
            del self._buffer[:FRAME_SIZE]

        return frames


def decode_frame(raw: bytes) -> ControllerFrame:
    if len(raw) != FRAME_SIZE:
        raise ValueError(f"frame must be {FRAME_SIZE} bytes, got {len(raw)}")
    if not raw.startswith(HEADER) or not raw.endswith(TRAILER):
        raise ValueError("frame markers are invalid")

    payload = raw[len(HEADER) : -len(TRAILER)]
    metadata_word = int.from_bytes(
        payload[:METADATA_SIZE], byteorder="big", signed=False
    )
    channel_data = payload[METADATA_SIZE:]
    counts = tuple(
        int.from_bytes(
            channel_data[offset : offset + CHANNEL_BYTES],
            byteorder="big",
            signed=True,
        )
        for offset in range(0, len(channel_data), CHANNEL_BYTES)
    )
    if len(counts) != CHANNEL_COUNT:
        raise ValueError(f"decoded {len(counts)} channels, expected {CHANNEL_COUNT}")

    return ControllerFrame(metadata_word=metadata_word, channel_counts=counts, raw=raw)
