"""Session inventory and window provenance around the canonical feature loader."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from ..dataset import (
    FeatureDataset, _read_discontinuities, _read_rows, _segment_bounds,
    load_feature_dataset,
)
from ..types import CHANNEL_COUNT
from ..session_metadata import BOOLEAN_FIELDS, OPTIONAL_SESSION_FIELDS, validate_session_conditions


CURRENT_LABELS = ("rest", "splayed", "fist", "wrist_up", "wrist_down")
LEGACY_MANIFEST_FIELDS = (
    "session_id", "path", "complete", "gesture_set", "subject_id",
    "recording_date", "notes", "use_for_baseline", "group_id",
    "label_timing_valid", "posture", "source", "value_unit", "sample_rate_hz",
)
V2_MANIFEST_FIELDS = ("protocol", "session_role", "posture_set", "repetitions_per_posture", "gesture_set_id")
MANIFEST_FIELDS = (LEGACY_MANIFEST_FIELDS
                   + tuple(field for field in OPTIONAL_SESSION_FIELDS if field not in LEGACY_MANIFEST_FIELDS)
                   + V2_MANIFEST_FIELDS)


@dataclass(frozen=True)
class Session:
    session_id: str
    path: Path
    labels: tuple[str, ...]
    group_id: str
    metadata: dict


@dataclass(frozen=True)
class WindowInfo:
    session_id: str
    trial_id: str
    group_id: str
    phase: str
    window_index: int
    start_sample: int
    end_sample: int  # exclusive
    start_time_s: float
    end_time_s: float


@dataclass(frozen=True)
class BenchmarkDataset:
    data: FeatureDataset
    windows: tuple[WindowInfo, ...]


@dataclass(frozen=True)
class RawRecording:
    session: Session
    rows: tuple[dict, ...]
    samples: np.ndarray
    segments: tuple[tuple[int, int], ...]


def load_raw_recordings(sessions: tuple[Session, ...]) -> tuple[RawRecording, ...]:
    """Cache original raw rows; corruption never writes to recording directories."""
    recordings = []
    for session in sessions:
        rows = _read_rows(session.path / "samples.csv")
        samples = np.array([[float(row[f"CH{ch}"]) for ch in range(CHANNEL_COUNT)] for row in rows])
        samples.setflags(write=False)
        bounds = tuple(_segment_bounds(rows, _read_discontinuities(session.path / "events.csv")))
        recordings.append(RawRecording(session, tuple(rows), samples, bounds))
    return tuple(recordings)


def segment_rng(seed: int, session_id: str, segment_number: int) -> np.random.Generator:
    """Stable segment substreams, independent of family selection or Python hash."""
    identity = hashlib.sha256(session_id.encode("utf-8")).digest()[:16]
    words = [int.from_bytes(identity[i:i + 4], "big") for i in range(0, 16, 4)]
    return np.random.default_rng(np.random.SeedSequence([seed, *words, segment_number]))


def corrupt_heldout_dataset(
    clean: BenchmarkDataset, recordings: tuple[RawRecording, ...], test_indices: np.ndarray,
    config: dict, corruption, seed: int,
) -> tuple[BenchmarkDataset, list[dict]]:
    """Overlay held-out raw segments, then call the unchanged canonical loader.

    Temporary CSVs are an input adapter, not a second preprocessing pipeline.
    Only test feature rows are replaced in the returned dataset. Full continuous
    filtering and the initial baseline RMS are preserved, including their coupling
    to other intervals. Training feature rows always remain the clean originals.
    """
    from .corruptions import signal_power

    test_indices = np.asarray(test_indices, dtype=int)
    heldout_groups = set(clean.data.groups[test_indices])
    features = clean.data.features.copy()
    audit = []
    with tempfile.TemporaryDirectory(prefix="emg_corruption_") as temporary:
        for number, recording in enumerate(recordings):
            session = recording.session
            selection = np.array([i for i in test_indices if clean.windows[i].session_id == session.session_id], dtype=int)
            if not len(selection):
                continue
            values = recording.samples.copy()
            for segment_number, (start, end) in enumerate(recording.segments):
                group = f"{session.path.resolve()}::segment_{segment_number}"
                if group not in heldout_groups:
                    continue
                original = recording.samples[start:end]
                corrupted = corruption.apply(original, float(config["sample_rate_hz"]),
                                             segment_rng(seed, session.session_id, segment_number))
                if corrupted.shape != original.shape or not np.all(np.isfinite(corrupted)):
                    raise ValueError("corruption returned invalid raw samples")
                values[start:end] = corrupted
                difference = corrupted - original
                audit.append({
                    "session_id": session.session_id, "segment_number": segment_number,
                    "group_id": group, "phase": recording.rows[start]["phase"],
                    "start_sample": int(recording.rows[start]["sample_index"]),
                    "end_sample": int(recording.rows[end - 1]["sample_index"]) + 1,
                    "sample_count": end - start, "seed": seed,
                    "signal_power_raw": signal_power(original).tolist(),
                    "signal_power_demeaned": signal_power(original - original.mean(axis=0)).tolist(),
                    "corruption_rms": np.sqrt(signal_power(difference)).tolist(),
                })
            directory = Path(temporary) / str(number)
            directory.mkdir()
            for name in ("metadata.json", "events.csv"):
                shutil.copyfile(session.path / name, directory / name)
            with (directory / "samples.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(recording.rows[0]))
                writer.writeheader()
                for row, sample in zip(recording.rows, values):
                    writer.writerow({**row, **{f"CH{ch}": format(sample[ch], ".17g") for ch in range(CHANNEL_COUNT)}})
            overlay = Session(session.session_id, directory, session.labels, session.group_id, session.metadata)
            processed = load_benchmark_dataset([overlay], config)
            original_indices = [i for i, window in enumerate(clean.windows) if window.session_id == session.session_id]
            if len(original_indices) != len(processed.windows):
                raise ValueError("corruption changed window count")
            for local_index, original_index in enumerate(original_indices):
                old, new = clean.windows[original_index], processed.windows[local_index]
                if (old.start_sample, old.end_sample, old.phase, old.trial_id,
                        str(clean.data.labels[original_index])) != (
                        new.start_sample, new.end_sample, new.phase, new.trial_id,
                        str(processed.data.labels[local_index])):
                    raise ValueError("corruption changed window provenance")
            local_positions = {original_index: i for i, original_index in enumerate(original_indices)}
            features[selection] = processed.data.features[[local_positions[i] for i in selection]]
    data = FeatureDataset(features, clean.data.labels, clean.data.groups)
    return BenchmarkDataset(data, clean.windows), audit


def _boolean(value: str, field: str) -> bool | None:
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    if not value:
        return None
    raise ValueError(f"invalid {field}: {value!r}; expected true/false/blank")


def write_manifest(recordings: Path, destination: Path) -> None:
    """Inventory recorded facts; missing metadata stays blank, never uses mtime."""
    records = []
    for metadata_path in sorted(recordings.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        conditions = validate_session_conditions({key: metadata[key] for key in OPTIONAL_SESSION_FIELDS if key in metadata})
        directory = metadata_path.parent
        rows = _read_rows(directory / "samples.csv")
        events = _read_rows(directory / "events.csv")
        observed_labels = {row["gesture"] for row in rows if row["phase"] != "prepare"}
        declared_labels = set(metadata.get("gestures", ()))
        # Metadata names active gestures; the calibration protocol has a rest baseline.
        if metadata.get("baseline_seconds", 0) > 0:
            declared_labels.add("rest")
        labels = sorted(declared_labels | observed_labels)
        complete = metadata.get("session_complete")
        notes = [conditions["notes"]] if conditions.get("notes") else []
        if not rows:
            notes.append("No samples; gesture set from declared calibration protocol")
        reconnects = sum(event["event"] == "stream_reconnect" for event in events)
        if reconnects:
            notes.append(f"{reconnects} stream_reconnect events; not necessarily unique gaps")
        if complete is None and any(row["event"] == "session_end" for row in events):
            complete = True
            notes.append("Completion inferred from session_end event")
        timing = metadata.get("label_timing_valid")
        date = metadata.get("recording_date", "")
        match = re.search(r"(?:^|_)(\d{8})_\d{6}$", directory.name)
        if not date and match:
            date = datetime.strptime(match[1], "%Y%m%d").date().isoformat()
            notes.append("Date from timestamp in session name; timezone unknown")
        eligible = complete is True and timing is not False and set(labels) == set(CURRENT_LABELS)
        if complete is not True:
            notes.append("Incomplete or completion unknown")
        if timing is False:
            notes.append("Invalid label timing")
        if set(labels) != set(CURRENT_LABELS):
            notes.append("Legacy gesture set; select explicitly with --gesture-set")
        destination.parent.mkdir(parents=True, exist_ok=True)
        records.append({
            "session_id": metadata.get("session_id", directory.name),
            "path": Path(os.path.relpath(directory, destination.parent)).as_posix(),
            "complete": "" if complete is None else str(complete).lower(),
            "gesture_set": "|".join(labels),
            "subject_id": metadata.get("subject_id", ""),
            "recording_date": date, "notes": "; ".join(notes),
            "use_for_baseline": str(eligible).lower(),
            "group_id": directory.relative_to(recordings).as_posix(),
            "label_timing_valid": "" if timing is None else str(timing).lower(),
            **{key: metadata.get(key, "") for key in
               ("posture", "source", "value_unit", "sample_rate_hz")},
            **{key: (str(conditions[key]).lower() if key in BOOLEAN_FIELDS and conditions.get(key) is not None
                      else conditions.get(key) or "")
               for key in OPTIONAL_SESSION_FIELDS if key not in LEGACY_MANIFEST_FIELDS},
            **{key: metadata.get(key, "") for key in ("protocol", "session_role", "repetitions_per_posture")},
            "posture_set": "|".join(metadata.get("posture_order", [])),
            "gesture_set_id": metadata.get("gesture_set", ""),
        })
    if not records:
        raise ValueError(f"no sessions found under {recordings}")
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def select_sessions(
    manifest: Path, labels: tuple[str, ...], requested: list[str] | None = None,
    metadata_filters: dict[str, str] | None = None,
) -> tuple[list[Session], list[dict[str, str]]]:
    records = _read_rows(manifest)
    if not records or not set(MANIFEST_FIELDS[:9]).issubset(records[0]):
        raise ValueError("manifest is empty or missing required columns")
    ids = [row["session_id"] for row in records]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("manifest session IDs must be nonempty and unique")
    if requested and set(requested) - set(ids):
        raise ValueError(f"unknown session IDs: {sorted(set(requested) - set(ids))}")
    if metadata_filters and set(metadata_filters) - set(MANIFEST_FIELDS):
        raise ValueError(f"unknown manifest filter fields: {sorted(set(metadata_filters) - set(MANIFEST_FIELDS))}")
    included, excluded = [], []
    seen_paths: set[Path] = set()
    for row in records:
        session_id = row["session_id"]
        reasons = []
        declared_labels = tuple(sorted(row["gesture_set"].split("|")))
        path = (manifest.parent / row["path"]).resolve()
        if requested and session_id not in requested:
            reasons.append("not selected")
        else:
            if _boolean(row["complete"], "complete") is not True:
                reasons.append("incomplete or completion unknown")
            if _boolean(row.get("label_timing_valid", ""), "label_timing_valid") is False:
                reasons.append("invalid label timing")
            if set(declared_labels) != set(labels):
                reasons.append("incompatible gesture set: " + ", ".join(declared_labels))
            if not reasons and not requested and _boolean(row["use_for_baseline"], "use_for_baseline") is not True:
                reasons.append("use_for_baseline=false; select explicitly to opt in")
        if reasons:
            excluded.append({"session_id": session_id, "reason": "; ".join(reasons)})
            continue
        if not row["path"] or any(not (path / name).is_file() for name in
                                  ("metadata.json", "samples.csv", "events.csv")):
            raise ValueError(f"{session_id}: missing metadata.json, samples.csv or events.csv at {path}")
        if path in seen_paths:
            raise ValueError(f"duplicate session path in manifest: {path}")
        seen_paths.add(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        # Nonempty manifest annotations override optional metadata only. Legacy gaps
        # stay unknown; provenance/labels/sample parameters cannot be overridden.
        conditions = {key: metadata.get(key) for key in OPTIONAL_SESSION_FIELDS}
        for key in OPTIONAL_SESSION_FIELDS:
            value = row.get(key, "")
            if value:
                conditions[key] = _boolean(value, key) if key in BOOLEAN_FIELDS else value
        conditions = validate_session_conditions(conditions)
        metadata = {**metadata, **conditions}
        filter_reasons = []
        for key, expected in (metadata_filters or {}).items():
            actual = metadata.get(key) if key in OPTIONAL_SESSION_FIELDS else row.get(key)
            if key in BOOLEAN_FIELDS:
                expected = _boolean(expected, key)
            if str(actual).lower() != str(expected).lower() or actual is None:
                filter_reasons.append(f"metadata filter {key}={expected} (actual={actual})")
        if filter_reasons:
            excluded.append({"session_id": session_id, "reason": "; ".join(filter_reasons)})
            continue
        if metadata.get("session_id", path.name) != session_id:
            raise ValueError(f"{session_id}: metadata session ID differs from manifest")
        if metadata.get("session_complete") is False:
            excluded.append({"session_id": session_id, "reason": "metadata marks session incomplete"})
            continue
        if metadata.get("label_timing_valid") is False or _boolean(
            row.get("label_timing_valid", ""), "label_timing_valid"
        ) is False:
            excluded.append({"session_id": session_id, "reason": "invalid label timing"})
            continue
        if metadata.get("session_complete") is None and not any(
            event["event"] == "session_end" for event in _read_rows(path / "events.csv")
        ):
            raise ValueError(f"{session_id}: completion cannot be verified")
        if metadata.get("channel_count") != CHANNEL_COUNT:
            raise ValueError(f"{session_id}: expected {CHANNEL_COUNT} channels")
        sample_rows = _read_rows(path / "samples.csv")
        actual_labels = {sample["gesture"] for sample in sample_rows if sample["phase"] != "prepare"}
        if actual_labels != set(labels) or set(metadata.get("gestures", ())) | {"rest"} != set(labels):
            raise ValueError(f"{session_id}: recorded labels disagree with manifest/gesture set")
        if not sample_rows:
            raise ValueError(f"{session_id}: empty samples.csv")
        indices = np.array([int(sample["sample_index"]) for sample in sample_rows])
        if np.any(np.diff(indices) != 1):
            raise ValueError(f"{session_id}: sample indices must be contiguous and ordered")
        values = np.array([[float(sample[f"CH{ch}"]) for ch in range(CHANNEL_COUNT)]
                           for sample in sample_rows])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{session_id}: non-finite channel values")
        if not any(sample["phase"] == "baseline" for sample in sample_rows):
            raise ValueError(f"{session_id}: missing rest baseline")
        included.append(Session(session_id, path, declared_labels, row["group_id"] or session_id, metadata))
    return included, excluded


def group_sessions_by_metadata(sessions: list[Session], fields: tuple[str, ...]) -> dict[tuple, list[Session]]:
    """Small grouping helper for future condition/session comparisons; no new split."""
    allowed = {"session_id", "group_id", "subject_id", "posture", "source", "value_unit", "sample_rate_hz",
               "protocol", "session_role", "repetitions_per_posture", *OPTIONAL_SESSION_FIELDS}
    if not fields or set(fields) - allowed:
        raise ValueError("grouping fields must be known manifest/metadata fields")
    groups = {}
    for session in sessions:
        values = tuple(session.session_id if key == "session_id" else
                       session.group_id if key == "group_id" else session.metadata.get(key)
                       for key in fields)
        groups.setdefault(values, []).append(session)
    return groups


def load_benchmark_dataset(sessions: list[Session], config: dict) -> BenchmarkDataset:
    """Import all signal processing; annotate exactly the windows it produces."""
    data = load_feature_dataset(
        [session.path for session in sessions],
        **{key: config[key] for key in (
            "window_size", "hop_size", "thresholds", "bandpass_hz", "guard_samples",
            "spatial_reference", "baseline_normalize",
        )},
    )
    if not np.all(np.isfinite(data.features)):
        raise ValueError("canonical preprocessing produced non-finite features")
    windows = []
    expected_labels, expected_groups = [], []
    guard, size, hop = config["guard_samples"], config["window_size"], config["hop_size"]
    for session in sessions:
        rows = _read_rows(session.path / "samples.csv")
        discontinuities = _read_discontinuities(session.path / "events.csv")
        window_index = 0
        for segment_number, (start, end) in enumerate(_segment_bounds(rows, discontinuities)):
            if rows[start]["phase"] == "prepare" or end - start <= 2 * guard:
                continue
            group = f"{session.path.resolve()}::segment_{segment_number}"
            for offset in range(start + guard, end - guard - size + 1, hop):
                sample_start = int(rows[offset]["sample_index"])
                fs = float(session.metadata["sample_rate_hz"])
                windows.append(WindowInfo(
                    session.session_id, rows[start]["trial"], group, rows[start]["phase"],
                    window_index, sample_start, sample_start + size,
                    float(rows[offset]["time_s"]), float(rows[offset + size - 1]["time_s"]) + 1 / fs,
                ))
                window_index += 1
                expected_labels.append(rows[start]["gesture"])
                expected_groups.append(group)
    if not np.array_equal(data.labels, expected_labels) or not np.array_equal(data.groups, expected_groups):
        raise ValueError("window provenance differs from canonical loader; benchmark needs updating")
    return BenchmarkDataset(data, tuple(windows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument("--output", type=Path, default=Path("data/session_manifest.csv"))
    args = parser.parse_args()
    write_manifest(args.recordings.resolve(), args.output.resolve())
    print(f"Manifest written to {args.output}")


if __name__ == "__main__":
    main()
