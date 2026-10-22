"""Dataset v2 protocol, recording orchestration, and strict recording validation.

This layer labels acquisition; signal processing remains in emg.dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import re
import time

import numpy as np

from .calibration import CalibrationConfig, CalibrationRunner
from .dataset import _read_rows
from .recording import SessionRecorder
from .session_metadata import OPTIONAL_SESSION_FIELDS, validate_session_conditions
from .sources import SyntheticSource
from .types import SAMPLE_RATE_HZ


V2_GESTURES = ("rest", "fist", "splayed", "wrist_up", "wrist_down",
               "radial_deviation", "ulnar_deviation", "pinch", "point")
FIVE_CORE_GESTURES = ("rest", "fist", "splayed", "wrist_up", "wrist_down")
V2_POSTURES = ("reference", "elbow_flexed", "elbow_extended", "arm_abducted")
V2_COMPACT_GESTURES = tuple(g for g in V2_GESTURES if g not in ("radial_deviation", "ulnar_deviation"))
V2_ROW_FIELDS = ("trial_id", "arm_posture", "repetition_index", "session_id",
                 "electrode_placement_id", "forearm_orientation", "shoulder_position",
                 "elbow_position", "dynamic_arm_motion")


class DatasetV2SyntheticSource(SyntheticSource):
    """Arbitrary new patterns for software checks only; no physiological claims."""
    _patterns = {**SyntheticSource._patterns,
                 "splayed": (.3, .45, .95, 1, .85, .4, .25, .2),
                 "wrist_up": (.2, .3, .4, .9, 1, .9, .3, .2),
                 "wrist_down": (.9, .8, .6, .2, .3, .2, .6, .8),
                 "radial_deviation": (.2, .8, .9, .3, .2, .4, .7, .3),
                 "ulnar_deviation": (.7, .2, .3, .7, .8, .2, .3, .8),
                 "point": (.5, .4, .8, .2, .4, .7, .3, .9)}


@dataclass(frozen=True)
class DatasetV2Protocol:
    session_role: str = "A"
    gestures: tuple[str, ...] = V2_GESTURES
    postures: tuple[str, ...] | None = None
    repetitions: int = 6
    seed: int = 7
    baseline_seconds: float = 5.0
    preparation_seconds: float = 3.0
    gesture_seconds: float = 3.0
    rest_seconds: float = 2.0
    independent_rest_seconds: float = 60.0
    posture_change_seconds: float = 15.0
    randomize_postures: bool = False
    block_conditions: dict | None = None
    posture_chunk: bool = False

    def __post_init__(self):
        if self.session_role not in ("A", "B", "C"):
            raise ValueError("session role must be A, B or C")
        postures = self.postures or (V2_POSTURES if self.session_role == "A" else ("reference",))
        object.__setattr__(self, "postures", tuple(postures))
        if (not self.gestures or "rest" not in self.gestures or len(self.gestures) < 2
                or len(set(self.gestures)) != len(self.gestures)):
            raise ValueError("Dataset v2 gestures must be unique and include rest and an active class")
        if any(not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in self.gestures):
            raise ValueError("gesture names must use letters, digits and underscores")
        if (not postures or (not self.posture_chunk and postures[0] != "reference")
                or len(set(postures)) != len(postures) or set(postures) - set(V2_POSTURES)):
            raise ValueError("use unique static Dataset v2 postures, with reference first unless recording a posture chunk")
        if self.repetitions < 2:
            raise ValueError("at least two repetitions per gesture are required")
        durations = (self.baseline_seconds, self.preparation_seconds, self.gesture_seconds,
                     self.rest_seconds, self.independent_rest_seconds, self.posture_change_seconds)
        if any(not np.isfinite(value) or value <= 0 for value in durations):
            raise ValueError("all phase durations must be positive and finite")
        if min(self.baseline_seconds, self.gesture_seconds, self.independent_rest_seconds) * SAMPLE_RATE_HZ < 101:
            raise ValueError("baseline, gesture and independent rest must provide guarded 25-sample windows (at least 0.808 s)")
        for posture, values in (self.block_conditions or {}).items():
            if posture not in postures or set(values) - {"shoulder_position", "elbow_position", "forearm_orientation", "notes"}:
                raise ValueError("block conditions may specify positions/orientation/notes for selected postures only")
            validate_session_conditions(values)

    def schedule(self, session_id: str) -> tuple[tuple[str, ...], list[dict]]:
        generator = random.Random(self.seed)
        remaining = list(self.postures if self.posture_chunk else self.postures[1:])
        if self.randomize_postures:
            generator.shuffle(remaining)
        postures = tuple(remaining) if self.posture_chunk else ("reference", *remaining)
        trials = []
        for posture in postures:
            for repetition in range(1, self.repetitions + 1):
                gestures = list(self.gestures)
                generator.shuffle(gestures)
                trials.extend({"trial_id": f"{session_id}_{posture}_{gesture}_r{repetition:02d}",
                               "gesture": gesture, "arm_posture": posture, "repetition_index": repetition}
                              for gesture in gestures)
        return postures, trials


def parse_v2_conditions(values: dict) -> tuple[dict, dict]:
    """Identity/preset fields are allowed only for the explicitly selected v2 protocol."""
    allowed = {*OPTIONAL_SESSION_FIELDS, "session_id", "subject_id", "gesture_set", "block_conditions",
               "collection_group_id", "collection_purpose"}
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError("unknown Dataset v2 conditions fields")
    conditions = validate_session_conditions({key: value for key, value in values.items() if key in OPTIONAL_SESSION_FIELDS})
    for key in ("session_id", "subject_id", "gesture_set", "collection_group_id", "collection_purpose"):
        if key in values and (not isinstance(values[key], str) or not values[key]):
            raise ValueError(f"{key} must be nonempty text")
        if key in values:
            conditions[key] = values[key]
    if conditions.get("collection_purpose", "training") not in ("training", "held_out_evaluation"):
        raise ValueError("collection_purpose must be training or held_out_evaluation")
    blocks = values.get("block_conditions", {})
    if not isinstance(blocks, dict):
        raise ValueError("block_conditions must be an object keyed by posture")
    if conditions.get("dynamic_arm_motion") not in (None, False):
        raise ValueError("Dataset v2 supports static arm conditions only")
    if conditions.get("arm_posture") not in (None, "reference"):
        raise ValueError("session starts in reference posture; use block_conditions for other positions")
    return conditions, blocks


def v2_metadata(session_id: str, subject_id: str, placement_id: str, protocol: DatasetV2Protocol,
                conditions: dict, source: str, value_unit: str) -> dict:
    for key, value in (("session_id", session_id), ("subject_id", subject_id), ("electrode_placement_id", placement_id)):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError(f"{key} must be a nonempty identifier using letters, digits, underscores or hyphens")
    if "electrode_reapplied" in conditions and conditions["electrode_reapplied"] is not None and conditions["electrode_reapplied"] != (protocol.session_role != "A"):
        raise ValueError("Session A keeps one application; B/C require a new application")
    order, planned = protocol.schedule(session_id)
    return {**conditions, "protocol": "dataset_v2", "format_version": 2,
            "session_role": protocol.session_role, "session_id": session_id, "subject_id": subject_id,
            "electrode_placement_id": placement_id, "electrode_reapplied": protocol.session_role != "A",
            "recording_date": conditions.get("recording_date") or datetime.now().astimezone().date().isoformat(),
            "recording_started_at": datetime.now().astimezone().isoformat(),
            "source": source, "value_unit": value_unit, "label_timing_valid": True,
            "gestures": list(protocol.gestures), "gesture_set": conditions.get("gesture_set") or (
                "dataset_v2_9class" if protocol.gestures == V2_GESTURES else "dataset_v2_custom"),
            "arm_posture": "reference", "posture": "multi_static" if len(order) > 1 else order[0],
            "forearm_orientation": conditions.get("forearm_orientation") or "neutral", "dynamic_arm_motion": False,
            "posture_chunk": protocol.posture_chunk, "baseline_posture": "reference",
            "posture_order": list(order), "planned_trial_order": planned, "trial_order": [], "trials": [],
            "rest_segments": [], "block_conditions": protocol.block_conditions or {},
            "trials_per_gesture": protocol.repetitions, "repetitions_per_posture": protocol.repetitions,
            "recording_seed": protocol.seed,
            **{name: getattr(protocol, name) for name in (
                "baseline_seconds", "preparation_seconds", "gesture_seconds", "rest_seconds",
                "independent_rest_seconds", "posture_change_seconds")}}


class DatasetV2Runner(CalibrationRunner):
    def __init__(self, source, recorder: SessionRecorder, protocol: DatasetV2Protocol, *,
                 announce=print, wait_for_prompts=True):
        super().__init__(source, recorder, CalibrationConfig(gestures=protocol.gestures,
                         trials_per_gesture=protocol.repetitions, baseline_seconds=protocol.baseline_seconds,
                         preparation_seconds=protocol.preparation_seconds, gesture_seconds=protocol.gesture_seconds,
                         rest_seconds=protocol.rest_seconds),
                         announce=announce, wait_for_prompts=wait_for_prompts)
        self.protocol = protocol

    def _context(self, posture, trial_id, repetition=None):
        metadata = self.recorder._metadata
        overrides = (self.protocol.block_conditions or {}).get(posture, {})
        self.recorder.conditions = {
            "trial_id": trial_id, "arm_posture": posture, "repetition_index": repetition,
            "session_id": metadata["session_id"], "electrode_placement_id": metadata["electrode_placement_id"],
            "dynamic_arm_motion": False,
            "forearm_orientation": overrides.get("forearm_orientation") or metadata["forearm_orientation"],
            **{key: overrides.get(key, metadata.get(key) if posture == "reference" else None)
               for key in ("shoulder_position", "elbow_position")}}

    def _transition(self, posture):
        self._context(posture, f"{self.recorder._metadata['session_id']}_{posture}_transition")
        self.announce(f"CHANGE POSTURE: {posture}. Keep electrodes in place; settle before the countdown ends.")
        if posture == "reference":
            notes = self.recorder._metadata.get("notes", "")
            reference_description = notes.split("Reference arm/position: ", 1)[-1] if "Reference arm/position: " in notes else "your original reference position"
            self.announce(f"POSITION: Return to {reference_description}. Use this position for the baseline and long rests.")
        description = (self.protocol.block_conditions or {}).get(posture, {}).get("notes")
        if description:
            self.announce(f"POSITION: {description}")
        self._record_phase("prepare", "rest", self.recorder.conditions["trial_id"], self.protocol.posture_change_seconds)

    def _long_rest(self, name):
        identity = f"{self.recorder._metadata['session_id']}_{name}"
        self._context("reference", identity)
        start, wall = self._next_index(), time.monotonic()
        self.announce(f"Independent {name}: remain relaxed in reference posture for {self.protocol.independent_rest_seconds:g} s.")
        self._record_phase(name, "rest", identity, self.protocol.independent_rest_seconds)
        record = {"trial_id": identity, "phase": name, "arm_posture": "reference",
                  "start_sample": start, "end_sample": self._next_index(),
                  "sample_duration_seconds": (self._next_index() - start) / SAMPLE_RATE_HZ,
                  "wall_duration_seconds": time.monotonic() - wall}
        self.recorder._metadata["rest_segments"].append(record)
        self.recorder.flush()
        self.recorder.update_metadata({})

    def run(self):
        metadata, protocol = self.recorder._metadata, self.protocol
        self._transition("reference")
        self._context("reference", f"{metadata['session_id']}_normalization_baseline")
        self.announce("Short normalization baseline: relax.")
        self._record_phase("baseline", "rest", self.recorder.conditions["trial_id"], protocol.baseline_seconds)
        self._long_rest("rest_pre")
        previous = "reference"
        for planned in metadata["planned_trial_order"]:
            posture, gesture, identity = planned["arm_posture"], planned["gesture"], planned["trial_id"]
            if posture != previous:
                self._transition(posture)
                previous = posture
            self._context(posture, identity, planned["repetition_index"])
            self._record_preparation(gesture, identity)
            self.announce(f"HOLD {gesture} | {posture} | repetition {planned['repetition_index']}/{protocol.repetitions}")
            start, wall = self._next_index(), time.monotonic()
            self._record_phase("gesture", gesture, identity, protocol.gesture_seconds)
            trial = {**planned, **self.recorder.conditions, "start_sample": start, "end_sample": self._next_index(),
                     "sample_duration_seconds": (self._next_index() - start) / SAMPLE_RATE_HZ,
                     "wall_duration_seconds": time.monotonic() - wall}
            metadata["trials"].append(trial)
            metadata["trial_order"].append(identity)
            self.recorder.flush()
            self.recorder.update_metadata({})
            self.announce("RELAX")
            self._record_phase("rest", "rest", identity, protocol.rest_seconds)
        if previous != "reference":
            self._transition("reference")
        self._long_rest("rest_post")
        self.recorder.write_event(self._next_index(), "session_end")
        self.recorder.update_metadata({"recording_finished_at": datetime.now().astimezone().isoformat()})


def validate_v2_recording(directory: Path, *, require_complete=True) -> dict:
    """Reject missing/short/overlapping/mislabeled trials before any evaluation."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    rows = _read_rows(directory / "samples.csv")
    events = _read_rows(directory / "events.csv")
    prefix = f"{metadata.get('session_id', directory.name)}: "
    def fail(message):
        raise ValueError(prefix + message)
    if (directory / "condition_corrections.json").exists():
        fail("physical-posture corrections are recorded in condition_corrections.json; apply them explicitly before evaluation (the original reference labels are not valid)")
    required = ("session_id", "subject_id", "electrode_placement_id", "recording_date", "session_role",
                "gestures", "posture_order", "repetitions_per_posture", "trial_order", "planned_trial_order", "trials")
    if metadata.get("protocol") != "dataset_v2" or any(not metadata.get(key) for key in required):
        fail("Dataset v2 metadata missing")
    if require_complete and metadata.get("session_complete") is not True:
        fail("recording incomplete; missing gestures/postures or short repetition counts may remain")
    if metadata.get("label_timing_valid") is not True:
        fail("invalid trial timing")
    validate_session_conditions({key: metadata[key] for key in OPTIONAL_SESSION_FIELDS if key in metadata})
    if metadata.get("dynamic_arm_motion") is not False:
        fail("Dataset v2 requires dynamic_arm_motion=false")
    if metadata.get("session_role") not in ("A", "B", "C") or metadata.get("electrode_reapplied") != (metadata["session_role"] != "A"):
        fail("invalid application/session role metadata")
    if metadata.get("channel_count") != 8 or metadata.get("sample_rate_hz") != SAMPLE_RATE_HZ:
        fail("expected 8 channels at 125 Hz")
    if not rows or not set(V2_ROW_FIELDS).issubset(rows[0]):
        fail("sample-level trial/condition metadata missing")
    if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
        fail("sample indices must be contiguous and start at zero")
    values = np.array([[float(row[f"CH{i}"]) for i in range(8)] for row in rows])
    if not np.all(np.isfinite(values)):
        fail("non-finite channel samples")
    if any(abs(float(row["time_s"]) - i / SAMPLE_RATE_HZ) > 1e-5 for i, row in enumerate(rows)):
        fail("invalid sample timing")
    gestures, postures, reps = metadata["gestures"], metadata["posture_order"], metadata["repetitions_per_posture"]
    if (len(gestures) < 2 or "rest" not in gestures or len(set(gestures)) != len(gestures)
            or (not metadata.get("posture_chunk", False) and postures[0] != "reference") or len(set(postures)) != len(postures)
            or set(postures) - set(V2_POSTURES) or not isinstance(reps, int) or reps < 2):
        fail("invalid gesture/posture/repetition protocol")
    expected = {(p, g, r) for p in postures for g in gestures for r in range(1, reps + 1)}
    trials = metadata["trials"]
    observed = [(t["arm_posture"], t["gesture"], t["repetition_index"]) for t in trials]
    if set(observed) != expected or len(observed) != len(expected):
        fail(f"gesture missing, posture incomplete, duplicate trials or repetition count short: missing={sorted(expected-set(observed))}")
    ids = [t["trial_id"] for t in trials]
    if len(set(ids)) != len(ids) or ids != metadata["trial_order"] or ids != [t["trial_id"] for t in metadata["planned_trial_order"]]:
        fail("duplicate trial IDs or actual/planned trial order disagreement")
    minimum = 101
    occupied = np.zeros(len(rows), dtype=bool)
    intervals = trials + metadata.get("rest_segments", [])
    if {t.get("phase") for t in metadata.get("rest_segments", [])} != {"rest_pre", "rest_post"} or len(metadata.get("rest_segments", [])) != 2:
        fail("both independent rest_pre and rest_post recordings are required")
    for trial in intervals:
        start, end = trial["start_sample"], trial["end_sample"]
        phase = trial.get("phase", "gesture")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(rows) or end-start < minimum:
            fail(f"impossible/short trial duration: {trial['trial_id']}")
        if np.any(occupied[start:end]):
            fail("overlapping trial labels")
        occupied[start:end] = True
        duration = (end-start)/SAMPLE_RATE_HZ
        requested = metadata["gesture_seconds"] if phase == "gesture" else metadata["independent_rest_seconds"]
        if abs(duration-requested) > 1/SAMPLE_RATE_HZ or abs(duration-trial["sample_duration_seconds"]) > 1e-8:
            fail(f"trial timing disagrees with configured/actual duration: {trial['trial_id']}")
        gesture = trial.get("gesture", "rest")
        if any(row["phase"] != phase or row["gesture"] != gesture or row["trial"] != trial["trial_id"]
               or row["trial_id"] != trial["trial_id"] or row["arm_posture"] != trial["arm_posture"] for row in rows[start:end]):
            fail(f"sample labels disagree with trial metadata: {trial['trial_id']}")
    if any(row["phase"] in ("gesture", "rest_pre", "rest_post") and not occupied[i] for i, row in enumerate(rows)):
        fail("unregistered trial/rest sample labels")
    if any(row["phase"] not in ("prepare", "baseline", "gesture", "rest", "rest_pre", "rest_post")
           or not row["trial_id"] or (row["phase"] in ("baseline", "rest", "rest_pre", "rest_post") and row["gesture"] != "rest")
           for row in rows):
        fail("unknown phase or invalid rest/trial labels")
    if any(not 0 <= int(event["sample_index"]) <= len(rows) for event in events):
        fail("event timing outside recording")
    for trial in intervals:
        phase = trial.get("phase", "gesture")
        for suffix, boundary in (("start", trial["start_sample"]), ("end", trial["end_sample"])):
            matches = [event for event in events if event["event"] == f"{phase}_{suffix}"
                       and event["trial"] == trial["trial_id"]]
            if len(matches) != 1 or int(matches[0]["sample_index"]) != boundary or matches[0]["trial_id"] != trial["trial_id"]:
                fail(f"event/trial timing disagreement: {trial['trial_id']}")
        if not np.isfinite(trial["wall_duration_seconds"]) or trial["wall_duration_seconds"] < 0:
            fail("invalid wall-clock duration")
        if phase == "gesture":
            if trial.get("session_id") != metadata["session_id"] or trial.get("electrode_placement_id") != metadata["electrode_placement_id"]:
                fail("trial/session placement identity mismatch")
            if any(row["repetition_index"] != str(trial["repetition_index"])
                   or any(row[key] != ("" if trial.get(key) is None else str(trial[key]))
                          for key in ("shoulder_position", "elbow_position", "forearm_orientation"))
                   for row in rows[trial["start_sample"]:trial["end_sample"]]):
                fail("sample-level repetition/position metadata disagrees with trial")
    if any(row["session_id"] != metadata["session_id"] or row["electrode_placement_id"] != metadata["electrode_placement_id"]
           or row["dynamic_arm_motion"] != "false" or row["arm_posture"] not in {*postures, "reference"}
           or row["forearm_orientation"] not in ("neutral", "pronated", "supinated", "custom") for row in rows):
        fail("session identity, same-placement or static condition violated")
    if any(row["arm_posture"] == "reference" and "reference" not in postures
           and row["phase"] in ("gesture", "rest") for row in rows):
        fail("reference may only be used for setup/baseline/independent rest in a non-reference chunk")
    baseline = [i for i,row in enumerate(rows) if row["phase"] == "baseline"]
    if (len(baseline) < minimum or baseline != list(range(baseline[0], baseline[-1]+1))
            or len(baseline) != round(metadata["baseline_seconds"]*SAMPLE_RATE_HZ)
            or any(rows[i]["arm_posture"] != "reference" for i in baseline)):
        fail("normalization baseline missing, fragmented, too short or wrongly positioned")
    if require_complete and (metadata.get("recorded_sample_count") != len(rows)
                             or not any(e["event"] == "session_end" and int(e["sample_index"]) == len(rows) for e in events)):
        fail("completion/sample-count mismatch")
    warnings = []
    if metadata["independent_rest_seconds"] < 45:
        warnings.append("independent rest is shorter than the recommended 45-60 seconds")
    if any(e["event"] == "stream_reconnect" for e in events):
        warnings.append("stream reconnect occurred; sample-derived duration does not include stream pauses")
    return {"session_id": metadata["session_id"], "trial_count": len(trials), "gesture_count": len(gestures),
            "postures": postures, "repetitions_per_posture": reps, "independent_rest_count": 2, "warnings": warnings}
