"""Prompted, sample-indexed calibration/enrolment workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from .recording import SessionRecorder
from .sources import SampleSource, SyntheticSource
from .types import SAMPLE_RATE_HZ


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    gestures: tuple[str, ...] = ("fist", "open_hand", "pinch")
    trials_per_gesture: int = 2
    baseline_seconds: float = 2.0
    preparation_seconds: float = 3.0
    gesture_seconds: float = 2.0
    rest_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.gestures:
            raise ValueError("at least one gesture is required")
        if self.trials_per_gesture <= 0:
            raise ValueError("trials_per_gesture must be positive")
        if min(
            self.baseline_seconds,
            self.preparation_seconds,
            self.gesture_seconds,
            self.rest_seconds,
        ) <= 0:
            raise ValueError("all phase durations must be positive")


class CalibrationRunner:
    TIMEOUTS_BEFORE_RECONNECT = 3
    MAX_RECONNECT_ATTEMPTS = 3

    def __init__(
        self,
        source: SampleSource,
        recorder: SessionRecorder,
        config: CalibrationConfig,
        *,
        announce: Callable[[str], None] = print,
        wait_for_prompts: bool = True,
    ) -> None:
        self.source = source
        self.recorder = recorder
        self.config = config
        self.announce = announce
        self.wait_for_prompts = wait_for_prompts

    def run(self) -> None:
        self.announce("Starting rest baseline.")
        self._record_phase("baseline", "rest", None, self.config.baseline_seconds)

        schedule = self._interleaved_schedule(
            self.config.gestures, self.config.trials_per_gesture
        )
        for trial_number, gesture in schedule:
            self._record_preparation(gesture, trial_number)
            self.announce(f"HOLD {gesture} (trial {trial_number})")
            self._record_phase(
                "gesture", gesture, trial_number, self.config.gesture_seconds
            )
            self.announce("RELAX")
            self._record_phase("rest", "rest", trial_number, self.config.rest_seconds)

        self.recorder.write_event(self._next_index(), "session_end")
        self.announce(f"Calibration complete: {self.recorder.sample_count} samples recorded.")

    @staticmethod
    def _interleaved_schedule(
        gestures: Sequence[str], trials_per_gesture: int
    ) -> list[tuple[int, str]]:
        # Deterministic interleaving avoids collecting all repetitions of one
        # gesture at the same point in the fatigue timeline.
        return [
            (trial, gesture)
            for trial in range(1, trials_per_gesture + 1)
            for gesture in gestures
        ]

    def _record_phase(
        self,
        phase: str,
        gesture: str,
        trial: int | None,
        duration_seconds: float,
    ) -> None:
        sample_count = round(duration_seconds * SAMPLE_RATE_HZ)
        start_index = self._next_index()
        self.recorder.write_event(start_index, f"{phase}_start", gesture, trial)

        if isinstance(self.source, SyntheticSource):
            self.source.set_gesture(gesture if phase == "gesture" else "rest")

        for _ in range(sample_count):
            sample = self._read_sample_resilient()
            self.recorder.write_sample(
                sample, phase=phase, gesture=gesture, trial=trial
            )

        self.recorder.write_event(
            self._next_index(), f"{phase}_end", gesture, trial
        )

    def _next_index(self) -> int:
        return self.recorder.sample_count

    def _record_preparation(self, gesture: str, trial: int) -> None:
        self.announce(f"Prepare: {gesture}")
        if not self.wait_for_prompts:
            return

        sample_count = round(self.config.preparation_seconds * SAMPLE_RATE_HZ)
        self.recorder.write_event(
            self._next_index(), "prepare_start", gesture, trial
        )
        last_number: int | None = None
        if isinstance(self.source, SyntheticSource):
            self.source.set_gesture("rest")

        for offset in range(sample_count):
            seconds_remaining = math.ceil((sample_count - offset) / SAMPLE_RATE_HZ)
            if seconds_remaining != last_number:
                self.announce(str(seconds_remaining))
                last_number = seconds_remaining
            sample = self._read_sample_resilient()
            self.recorder.write_sample(
                sample, phase="prepare", gesture=gesture, trial=trial
            )

        self.recorder.write_event(
            self._next_index(), "prepare_end", gesture, trial
        )

    def _read_sample_resilient(self):
        timeout_count = 0
        reconnect_attempts = 0
        stream_was_paused = False
        while True:
            try:
                sample = self.source.read_sample()
            except TimeoutError:
                timeout_count += 1
                stream_was_paused = True
                if timeout_count == 1:
                    self.announce(
                        "WARNING: stream paused; keep the current pose while reconnecting."
                    )
                reconnect = getattr(self.source, "reconnect", None)
                if (
                    timeout_count >= self.TIMEOUTS_BEFORE_RECONNECT
                    and callable(reconnect)
                    and reconnect_attempts < self.MAX_RECONNECT_ATTEMPTS
                ):
                    reconnect_attempts += 1
                    self.recorder.write_event(self._next_index(), "stream_reconnect")
                    self.announce(
                        f"Reopening serial port (attempt {reconnect_attempts}/"
                        f"{self.MAX_RECONNECT_ATTEMPTS})..."
                    )
                    reconnect()
                    timeout_count = 0
                    continue
                if reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
                    raise TimeoutError(
                        "controller stream did not resume after automatic port reopening"
                    )
                continue

            if stream_was_paused:
                self.announce("Stream resumed; continuing the current phase.")
            return sample
