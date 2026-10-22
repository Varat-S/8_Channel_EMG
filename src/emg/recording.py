"""Lossless session recording with sample-indexed event labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .types import CHANNEL_COUNT, SAMPLE_RATE_HZ, Sample


class SessionRecorder:
    """Write raw samples and event boundaries into a new session directory."""

    def __init__(self, directory: str | Path, metadata: dict[str, Any], *,
                 condition_fields: tuple[str, ...] = ()) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)

        self._samples_file = (self.directory / "samples.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self._events_file = (self.directory / "events.csv").open(
            "w", encoding="utf-8", newline=""
        )
        self._sample_writer = csv.writer(self._samples_file)
        self._event_writer = csv.writer(self._events_file)
        self.condition_fields = condition_fields
        self.conditions: dict[str, Any] = {}

        self._sample_writer.writerow(
            [
                "sample_index",
                "time_s",
                "frame_word",
                "phase",
                "gesture",
                "trial",
                *(f"CH{channel}" for channel in range(CHANNEL_COUNT)),
                *condition_fields,
            ]
        )
        self._event_writer.writerow(["sample_index", "event", "gesture", "trial", *condition_fields])

        complete_metadata = {
            "format_version": 1,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channel_count": CHANNEL_COUNT,
            "session_complete": False,
            **metadata,
        }
        self._metadata = complete_metadata
        self._write_metadata()
        self.sample_count = 0

    def write_sample(
        self,
        sample: Sample,
        *,
        phase: str,
        gesture: str,
        trial: int | str | None,
    ) -> None:
        self._sample_writer.writerow(
            [
                sample.index,
                f"{sample.index / SAMPLE_RATE_HZ:.6f}",
                "" if sample.frame_word is None else sample.frame_word,
                phase,
                gesture,
                "" if trial is None else trial,
                *(f"{value:.12g}" for value in sample.values),
                *self._condition_values(),
            ]
        )
        self.sample_count += 1

    def write_event(
        self,
        sample_index: int,
        event: str,
        gesture: str = "",
        trial: int | str | None = None,
    ) -> None:
        self._event_writer.writerow(
            [sample_index, event, gesture, "" if trial is None else trial, *self._condition_values()]
        )

    def _condition_values(self) -> list:
        return [str(value).lower() if isinstance(value, bool) else "" if value is None else value
                for value in (self.conditions.get(key) for key in self.condition_fields)]

    def update_metadata(self, values: dict[str, Any]) -> None:
        """Persist protocol progress, retaining incomplete status until finalization."""
        self._metadata.update(values)
        self._write_metadata()

    def close(self) -> None:
        self._samples_file.close()
        self._events_file.close()

    def flush(self) -> None:
        self._samples_file.flush()
        self._events_file.flush()

    def mark_complete(self) -> None:
        self._metadata["session_complete"] = True
        self._metadata["recorded_sample_count"] = self.sample_count
        self._write_metadata()

    def _write_metadata(self) -> None:
        (self.directory / "metadata.json").write_text(
            json.dumps(self._metadata, indent=2) + "\n", encoding="utf-8"
        )

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
