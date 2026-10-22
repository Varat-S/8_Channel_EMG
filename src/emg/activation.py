"""Stable-gesture detection and deliberate wake-sequence gating."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class StableGestureDetector:
    """Emit a gesture once after it remains confident for a minimum duration."""

    hold_seconds: float = 0.4
    confidence_threshold: float = 0.55
    _candidate: str | None = field(default=None, init=False)
    _candidate_since: float = field(default=0.0, init=False)
    _emitted: bool = field(default=False, init=False)

    def update(self, gesture: str, confidence: float, now: float) -> str | None:
        if confidence < self.confidence_threshold:
            self.reset()
            return None
        if gesture != self._candidate:
            self._candidate = gesture
            self._candidate_since = now
            self._emitted = False
            return None
        if not self._emitted and now - self._candidate_since >= self.hold_seconds:
            self._emitted = True
            return gesture
        return None

    def reset(self) -> None:
        self._candidate = None
        self._candidate_since = 0.0
        self._emitted = False


@dataclass(slots=True)
class WakeSequenceGate:
    """Arm output only after a sequence of stable, deliberate gestures.

    Neutral labels such as ``rest`` are separators: while locked they neither
    advance nor erase valid wake progress. Other wrong active gestures still
    reset the sequence, and the maximum gap between accepted steps still applies.
    """

    sequence: tuple[str, ...]
    max_step_gap_seconds: float = 2.5
    ignored_gestures: frozenset[str] = field(
        default_factory=lambda: frozenset({"rest"})
    )
    progress: int = field(default=0, init=False)
    armed: bool = field(default=False, init=False)
    _last_step_at: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.sequence:
            raise ValueError("wake sequence cannot be empty")

    @property
    def expected(self) -> str | None:
        return None if self.armed else self.sequence[self.progress]

    def accept(self, gesture: str, now: float) -> bool:
        """Accept one stable gesture event and return whether arming occurred."""

        if self.armed:
            return False
        if gesture in self.ignored_gestures:
            return False
        if (
            self._last_step_at is not None
            and now - self._last_step_at > self.max_step_gap_seconds
        ):
            self.progress = 0

        if gesture == self.sequence[self.progress]:
            self.progress += 1
            self._last_step_at = now
            if self.progress == len(self.sequence):
                self.armed = True
                return True
        elif gesture == self.sequence[0]:
            self.progress = 1
            self._last_step_at = now
        else:
            self.progress = 0
            self._last_step_at = None
        return False

    def disarm(self) -> None:
        self.armed = False
        self.progress = 0
        self._last_step_at = None
