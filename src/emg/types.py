"""Shared, dependency-free data types."""

from __future__ import annotations

from dataclasses import dataclass


CHANNEL_COUNT = 8
# The lab GUI is configured for 250 Hz, but direct serial measurements on
# 2026-09-14 show a clean nominal output rate of 125 complete frames/s.
CONTROLLER_CONFIGURED_RATE_HZ = 250
SAMPLE_RATE_HZ = 125


@dataclass(frozen=True, slots=True)
class Sample:
    """One simultaneous observation from all eight channels.

    ``values`` are deliberately not called volts. Live frames currently yield ADC
    counts because the controller's scaling has not yet been confirmed. Replay and
    synthetic sources can use other units, recorded in the session metadata.
    """

    index: int
    values: tuple[float, ...]
    frame_word: int | None = None

    def __post_init__(self) -> None:
        if len(self.values) != CHANNEL_COUNT:
            raise ValueError(f"expected {CHANNEL_COUNT} channels, got {len(self.values)}")
