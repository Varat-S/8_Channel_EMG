"""Audited compact recordings, matched causal filtering, and trial holdouts.

The historical decoder and benchmark modules remain independent of this path.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import iirnotch, sosfilt, sosfilt_zi, tf2sos

from .dataset import FeatureDataset, _read_rows, _segment_bounds
from .dataset_v2 import V2_COMPACT_GESTURES, validate_v2_recording
from .evaluation.benchmark import DEFAULT_CONFIG, validate_split
from .evaluation.dataset import BenchmarkDataset, WindowInfo
from .features import extract_time_domain_features
from .preprocessing import apply_spatial_reference, design_bandpass


def file_hashes(path):
    return {name: hashlib.sha256((path/name).read_bytes()).hexdigest()
            for name in ("metadata.json", "samples.csv", "events.csv")}


class CompactFilter:
    """Identical causal SOS implementation for batch training and live samples."""
    def __init__(self, notch_hz=None, notch_q=30.0):
        sections = [design_bandpass(125, 10, 55)]
        if notch_hz is not None:
            b, a = iirnotch(float(notch_hz), float(notch_q), fs=125)
            sections.append(tf2sos(b, a))
        self.sos = np.vstack(sections)
        self.zi = None

    def process(self, values):
        values = np.asarray(values, dtype=float)
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[:, :, None] * values[0][None, None, :]
        filtered, self.zi = sosfilt(self.sos, values, axis=0, zi=self.zi)
        return filtered

    def update(self, value):
        return self.process(np.asarray(value)[None, :])[0]


@dataclass
class CompactRecording:
    path: Path
    metadata: dict
    rows: list
    values: np.ndarray
    discontinuities: set
    trials: list
    audit: dict


def read_recording(
    path,
    *,
    reference_index=None,
    purpose="training",
    expected_gestures=V2_COMPACT_GESTURES,
):
    path = Path(path).resolve()
    metadata = json.loads((path/"metadata.json").read_text(encoding="utf-8"))
    hashes = file_hashes(path)
    if metadata.get("collection_purpose", "training") != purpose:
        raise ValueError(f"{path.name}: wrong collection purpose; evaluation data must never train the model")
    if metadata.get("label_timing_valid") is not True or metadata.get("sample_rate_hz") != 125 or metadata.get("channel_count") != 8:
        raise ValueError(f"{path.name}: incompatible or invalid acquisition")
    corrections_path = path/"condition_corrections.json"
    corrections = json.loads(corrections_path.read_text()) if corrections_path.exists() else None
    if corrections and (corrections.get("source_sha256") != hashes or corrections.get("session_id") != metadata["session_id"]):
        raise ValueError(f"{path.name}: condition-correction provenance no longer matches raw files")
    exclusions_path = path/"trial_exclusions.json"
    trial_exclusions = json.loads(exclusions_path.read_text()) if exclusions_path.exists() else None
    if trial_exclusions and (
        trial_exclusions.get("source_sha256") != hashes
        or trial_exclusions.get("session_id") != metadata["session_id"]
    ):
        raise ValueError(f"{path.name}: trial-exclusion provenance no longer matches raw files")
    if reference_index is not None:
        if hashes != reference_index["reference_source_sha256"]:
            raise ValueError("interrupted reference source has changed since its completed-trial audit")
        chosen_ids = {t["trial_id"] for t in reference_index["completed_reference_trials"]}
        trials = [dict(t) for t in metadata["trials"] if t["trial_id"] in chosen_ids]
        if {t["trial_id"] for t in trials} != chosen_ids:
            raise ValueError("audited reference trials no longer match saved metadata")
    else:
        if not metadata.get("session_complete"):
            raise ValueError(f"{path.name}: incomplete chunk")
        if not corrections:
            validate_v2_recording(path)
        trials = [dict(t) for t in metadata["trials"]]
        expected_gestures = tuple(expected_gestures)
        if tuple(metadata.get("gestures", ())) != expected_gestures:
            raise ValueError(
                f"{path.name}: gesture vocabulary/order differs from the expected protocol"
            )
        expected = {(p, g, rep) for p in metadata["posture_order"] for g in expected_gestures
                    for rep in range(1, metadata["repetitions_per_posture"]+1)}
        if {(t["arm_posture"], t["gesture"], t["repetition_index"]) for t in trials} != expected or len(trials) != len(expected):
            raise ValueError(f"{path.name}: missing/duplicate gesture repetitions")
    rows = _read_rows(path/"samples.csv")
    events = _read_rows(path/"events.csv")
    if [int(r["sample_index"]) for r in rows] != list(range(len(rows))):
        raise ValueError(f"{path.name}: noncontiguous sample indices")
    if any(abs(float(r["time_s"])-i/125) > 1e-5 for i,r in enumerate(rows)):
        raise ValueError("bad sample-derived timing")
    values = np.array([[float(r[f"CH{i}"]) for i in range(8)] for r in rows])
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite raw channel data")
    if any(r["session_id"] != metadata["session_id"] or r["electrode_placement_id"] != metadata["electrode_placement_id"]
           or r["dynamic_arm_motion"] != "false" for r in rows):
        raise ValueError("sample identity or static condition mismatch")
    occupied = np.zeros(len(rows), dtype=bool)
    for trial in trials:
        start, end = trial["start_sample"], trial["end_sample"]
        if not 0 <= start < end <= len(rows) or end-start != round(metadata["gesture_seconds"]*125):
            raise ValueError("incomplete registered gesture hold")
        if np.any(occupied[start:end]):
            raise ValueError("overlapping gesture holds")
        occupied[start:end] = True
        if any(r["phase"] != "gesture" or r["trial_id"] != trial["trial_id"] or r["gesture"] != trial["gesture"]
               or r["arm_posture"] != trial["arm_posture"] or int(r["repetition_index"]) != trial["repetition_index"]
               for r in rows[start:end]):
            raise ValueError("registered gesture does not match sample labels")
        for boundary, index in (("start",start),("end",end)):
            matched = [e for e in events if e["event"] == f"gesture_{boundary}" and e["trial"] == trial["trial_id"]]
            if not matched and reference_index is not None and metadata.get("session_complete") is False:
                continue  # audited metadata/sample boundaries survive the interrupted event flush
            if len(matched) != 1 or int(matched[0]["sample_index"]) != index:
                raise ValueError("gesture event boundary mismatch")
    if corrections:
        for override in corrections["sample_condition_overrides"]:
            start,end = override["start_sample"],override["end_sample"]
            if not 0 <= start < end <= len(rows):
                raise ValueError("invalid correction interval")
            for row in rows[start:end]:
                if row["arm_posture"] != override["recorded_arm_posture"] or row["phase"] != override["phase"]:
                    raise ValueError("correction does not match original condition labels")
                row["arm_posture"] = override["actual_arm_posture"]
    breaks = {int(e["sample_index"]) for e in events if e["event"] == "stream_reconnect"}
    if any(not 0 <= i <= len(rows) for i in breaks):
        raise ValueError("reconnect index outside recording")
    explicit_exclusions = {}
    if trial_exclusions:
        entries = trial_exclusions.get("excluded_trials")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{path.name}: trial exclusions must be a nonempty list")
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("trial_id"), str)
                or not isinstance(entry.get("reason"), str)
                or not entry["reason"]
                or entry["trial_id"] in explicit_exclusions
            ):
                raise ValueError(f"{path.name}: malformed or duplicate trial exclusion")
            explicit_exclusions[entry["trial_id"]] = entry["reason"]
        unknown = set(explicit_exclusions) - {trial["trial_id"] for trial in trials}
        if unknown:
            raise ValueError(f"{path.name}: exclusions reference unknown trials: {sorted(unknown)}")
    excluded, retained = [], []
    for trial in trials:
        start,end = trial["start_sample"],trial["end_sample"]
        reason = explicit_exclusions.get(trial["trial_id"])
        reason = reason or ("reconnect interrupted hold" if any(start <= i < end for i in breaks) else None)
        if trial["wall_duration_seconds"] > 1.5*metadata["gesture_seconds"]:
            reason = reason or "hold wall time exceeds nominal duration by more than 50 percent"
        if reason:
            excluded.append({"trial_id":trial["trial_id"],"reason":reason})
        else:
            retained.append(trial)
    audit = {"session_id":metadata["session_id"],"source_path":str(path),"source_sha256":hashes,
             "source_complete":metadata["session_complete"],"selected_trials":len(retained),"excluded_trials":excluded,
             "condition_corrections":corrections,"trial_exclusions":trial_exclusions,"sample_count":len(rows),
             "baseline_actual_posture":corrections["actual_baseline_posture"] if corrections else metadata.get("baseline_posture","reference")}
    return CompactRecording(path,metadata,rows,values,breaks,retained,audit)


def read_collection(collection_path):
    collection_path = Path(collection_path).resolve()
    collection = json.loads(collection_path.read_text())
    recordings = [read_recording(collection["reference_source"],reference_index=collection)]
    root = recordings[0].path.parent
    for chunk in collection["chunks"]:
        if chunk["purpose"] != "training":
            continue
        conditions = json.loads((collection_path.parent/chunk["conditions_file"]).read_text())
        recordings.append(read_recording(root/conditions["session_id"]))
    for recording in recordings:
        for key in ("electrode_placement_id","subject_id","value_unit","source","recording_date"):
            if recording.metadata[key] != recordings[0].metadata[key]:
                raise ValueError(f"cannot pool different {key} in a same-application model")
    return collection,recordings


def filtered_recording(recording, notch_hz):
    referenced = apply_spatial_reference(recording.values,"common_average")
    result = np.empty_like(referenced)
    boundaries = [0,*sorted(i for i in recording.discontinuities if 0<i<len(referenced)),len(referenced)]
    for start,end in zip(boundaries,boundaries[1:]):
        result[start:end] = CompactFilter(notch_hz).process(referenced[start:end])
    return result


def make_dataset(
    recordings,
    *,
    notch_hz=None,
    normalization="session_baseline",
    fixed_scale=None,
    include_signal_windows=False,
):
    filtered = [filtered_recording(r,notch_hz) for r in recordings]
    if normalization not in ("session_baseline","fixed_reference"):
        raise ValueError("unknown normalization")
    if fixed_scale is None:
        baseline = np.array([r["phase"]=="baseline" for r in recordings[0].rows])
        if sum(baseline) != 625:
            raise ValueError("reference baseline must contain all five seconds")
        fixed_scale = np.sqrt(np.mean(filtered[0][baseline]**2,axis=0))
    fixed_scale = np.asarray(fixed_scale,dtype=float)
    if fixed_scale.shape != (8,) or not np.all(np.isfinite(fixed_scale)) or np.any(fixed_scale<=1e-8):
        raise ValueError("invalid fixed reference scale")
    features,labels,groups,windows,conditions,signal_windows = [],[],[],[],[],[]
    for recording,values in zip(recordings,filtered):
        baseline = np.array([r["phase"]=="baseline" for r in recording.rows])
        if sum(baseline) != 625:
            raise ValueError("normalization baseline incomplete")
        scale = np.sqrt(np.mean(values[baseline]**2,axis=0)) if normalization=="session_baseline" else fixed_scale
        if np.any(scale<=1e-8):
            raise ValueError("zero normalization scale")
        values = values/scale
        retained_ids = {t["trial_id"] for t in recording.trials}
        for segment,(start,end) in enumerate(_segment_bounds(recording.rows,recording.discontinuities)):
            row=recording.rows[start]
            if row["phase"] not in ("gesture","rest_pre","rest_post"):
                continue
            if row["phase"]=="gesture" and row["trial_id"] not in retained_ids:
                continue
            group=f"{recording.metadata['session_id']}::segment_{segment}"
            for offset in range(start+38,end-38-25+1,5):
                signal_window=values[offset:offset+25]
                features.append(extract_time_domain_features(signal_window))
                if include_signal_windows:
                    signal_windows.append(signal_window.T.astype(np.float32,copy=False))
                labels.append(row["gesture"]);groups.append(row["trial_id"])
                windows.append(WindowInfo(recording.metadata["session_id"],row["trial_id"],group,row["phase"],
                                          len(windows),offset,offset+25,offset/125,(offset+25)/125))
                conditions.append({"arm_posture":row["arm_posture"],"forearm_orientation":row["forearm_orientation"]})
        available={w.trial_id for w in windows if w.session_id==recording.metadata["session_id"] and w.phase=="gesture"}
        if available != retained_ids:
            raise ValueError("some complete trials produced no guarded feature windows")
    dataset=BenchmarkDataset(FeatureDataset(np.asarray(features),np.asarray(labels),np.asarray(groups)),tuple(windows))
    if include_signal_windows:
        raw=np.asarray(signal_windows,dtype=np.float32)
        if raw.shape != (len(dataset.windows),8,25) or not np.all(np.isfinite(raw)):
            raise ValueError("invalid aligned signal windows")
        return dataset,conditions,fixed_scale,raw
    return dataset,conditions,fixed_scale


def compact_split(dataset,seed):
    """One held-out whole trial of every class in every recorded condition."""
    generator=np.random.default_rng(seed);test_ids=set()
    sessions=sorted({w.session_id for w in dataset.windows})
    for session in sessions:
        for label in V2_COMPACT_GESTURES:
            ids=sorted({w.trial_id for i,w in enumerate(dataset.windows) if w.phase=="gesture"
                        and w.session_id==session and dataset.data.labels[i]==label})
            if len(ids)<2:
                raise ValueError(f"{session}/{label}: fewer than two clean complete trials")
            test_ids.add(str(generator.choice(ids)))
    gesture=np.array([w.phase=="gesture" for w in dataset.windows])
    selected=np.isin(dataset.data.groups,list(test_ids))
    train=np.flatnonzero(gesture & ~selected);test=np.flatnonzero(gesture & selected)
    validate_split(dataset,train,test)
    return train,test
