"""Prepare short seven-class chunks alongside an audited, interrupted reference block.

Does not open hardware, change original recordings, or label an interrupted
recording complete. The reference index is provenance for subsequent analysis.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from emg.dataset_v2 import (DatasetV2Protocol, V2_COMPACT_GESTURES,
                           parse_v2_conditions, v2_metadata)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def audit_reference(directory):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("protocol") != "dataset_v2" or metadata.get("label_timing_valid") is not True:
        raise ValueError("a Dataset v2 recording with valid timing is required")
    if metadata.get("channel_count") != 8 or metadata.get("sample_rate_hz") != 125:
        raise ValueError("expected eight channels at nominal 125 Hz")
    with (directory / "samples.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (directory / "events.csv").open(encoding="utf-8", newline="") as stream:
        events = list(csv.DictReader(stream))
    if [int(r["sample_index"]) for r in rows] != list(range(len(rows))):
        raise ValueError("sample indices are not contiguous")
    if any(not math.isfinite(float(r[f"CH{channel}"])) for r in rows for channel in range(8)):
        raise ValueError("non-finite channel data")
    trials = [t for t in metadata["trials"] if t["arm_posture"] == "reference"
              and t["gesture"] in V2_COMPACT_GESTURES]
    if len({t["trial_id"] for t in trials}) != len(trials):
        raise ValueError("duplicate reference trial IDs")
    for gesture in V2_COMPACT_GESTURES:
        if sum(t["gesture"] == gesture for t in trials) < 2:
            raise ValueError(f"{gesture}: insufficient completed reference trials")
    for trial in trials:
        start, end = trial["start_sample"], trial["end_sample"]
        if not 0 <= start < end <= len(rows) or end - start < 101:
            raise ValueError("incomplete or short registered trial")
        if end - start != round(metadata["gesture_seconds"] * metadata["sample_rate_hz"]):
            raise ValueError("trial duration disagrees with requested hold")
        verified_events = 0
        for boundary, index in (("start", start), ("end", end)):
            matches = [e for e in events if e["event"] == f"gesture_{boundary}" and e["trial"] == trial["trial_id"]]
            if not matches and metadata.get("session_complete") is False:
                continue
            if len(matches) != 1 or int(matches[0]["sample_index"]) != index:
                raise ValueError("trial/event boundary mismatch")
            verified_events += 1
        if any(r["trial_id"] != trial["trial_id"] or r["gesture"] != trial["gesture"]
               or r["phase"] != "gesture" or r["arm_posture"] != "reference"
               or r["electrode_placement_id"] != metadata["electrode_placement_id"]
               or r["session_id"] != metadata["session_id"] or r["dynamic_arm_motion"] != "false"
               for r in rows[start:end]):
            raise ValueError("trial/sample label mismatch")
        trial["boundary_evidence"] = "metadata_samples_and_events" if verified_events == 2 else "metadata_and_samples_only"
    baseline = [r for r in rows if r["phase"] == "baseline"]
    if len(baseline) != round(metadata["baseline_seconds"] * metadata["sample_rate_hz"]):
        raise ValueError("reference baseline incomplete")
    if any(r["arm_posture"] != "reference" for r in baseline):
        raise ValueError("normalization baseline in wrong posture")
    return metadata, trials


def prepare(reference, output):
    metadata, trials = audit_reference(reference)
    output.mkdir(parents=True, exist_ok=False)
    group = output.name
    # Descriptions are acquisition instructions; angles are deliberately unspecified.
    positions = [
        ("extended", "elbow_extended", "neutral", 3, "training",
         "Left arm down beside your body, elbow comfortably nearly straight; thumb points forward. Keep still."),
        ("horizontal", "elbow_flexed", "pronated", 3, "training",
         "Left elbow bent beside your body, forearm horizontal in front of you, palm facing the floor. Keep still."),
        ("side", "arm_abducted", "supinated", 3, "training",
         "Left arm held comfortably away from your torso to the side, palm facing upward. Use a comfortable height and keep still."),
        ("unfamiliar", "arm_abducted", "neutral", 2, "held_out_evaluation",
         "Choose a comfortable diagonal position between forward and sideways, at a different height from the training positions. Thumb points upward; keep still. This chunk is for evaluation only."),
    ]
    chunks = []
    for number, (name, posture, orientation, reps, purpose, description) in enumerate(positions, 1):
        identity = f"{group}_{number}_{name}"
        conditions = {
            "session_id": identity, "subject_id": metadata["subject_id"],
            "electrode_placement_id": metadata["electrode_placement_id"], "electrode_reapplied": False,
            "arm_posture": "reference", "forearm_orientation": metadata["forearm_orientation"],
            "dynamic_arm_motion": False, "recording_date": None,
            "gesture_set": "dataset_v2_7class", "collection_group_id": group, "collection_purpose": purpose,
            "notes": ("Same electrode application as " + metadata["session_id"] +
                      ". Reference arm/position: Left arm; forearm perpendicular to the ground. "
                      "Return to this reference for baseline and independent rest. Angles not measured."),
            "block_conditions": {posture: {"forearm_orientation": orientation, "notes": description}},
        }
        validated, blocks = parse_v2_conditions(conditions)
        protocol = DatasetV2Protocol(gestures=V2_COMPACT_GESTURES, postures=(posture,),
                                     posture_chunk=True, repetitions=reps, seed=7+number, block_conditions=blocks)
        plan = v2_metadata(identity, metadata["subject_id"], metadata["electrode_placement_id"],
                           protocol, validated, "serial", "pending")
        count = len(plan["planned_trial_order"])
        if count >= 100:
            raise ValueError("chunk trial limit exceeded")
        config_path = output / (identity + ".conditions.json")
        write_json(config_path, conditions)
        write_json(output / (identity + ".plan.json"), plan)
        chunks.append({"number": number, "name": name, "conditions_file": config_path.name,
                       "posture": posture, "repetitions": reps, "seed": 7+number,
                       "purpose": purpose, "trials": count, "position_instructions": description})
    manifest = {
        "collection_group_id": group, "gestures": list(V2_COMPACT_GESTURES), "max_trials_per_recording": 99,
        "reference_source": str(reference.resolve()), "reference_source_complete": metadata.get("session_complete"),
        "reference_source_sha256": {name: hashlib.sha256((reference/name).read_bytes()).hexdigest()
                                    for name in ("metadata.json", "samples.csv", "events.csv")},
        "completed_reference_trial_count": len(trials), "completed_reference_trials": trials,
        "reference_trials_missing_event_boundaries": [t["trial_id"] for t in trials
                                                     if t["boundary_evidence"] == "metadata_and_samples_only"],
        "excluded_gestures": ["radial_deviation", "ulnar_deviation"],
        "reference_missing_rest_post": not any(s["phase"] == "rest_post" for s in metadata.get("rest_segments", [])),
        "new_training_trials": sum(c["trials"] for c in chunks if c["purpose"] == "training"),
        "new_evaluation_trials": sum(c["trials"] for c in chunks if c["purpose"] == "held_out_evaluation"),
        "chunks": chunks,
        "evaluation_policy": ("Split whole trials, never overlapping windows. Reserve the unfamiliar-position chunk "
                              "from training and tuning. Pool reference and training chunks only after an explicit "
                              "partial-recording adapter verifies source hashes and complete-trial boundaries. "
                              "The old full-session benchmark rejects the interrupted source; this index does not bypass it."),
    }
    write_json(output / "collection.json", manifest)
    print(json.dumps({"collection": str(output), "reference_trials": len(trials),
                      "new_training_trials": manifest["new_training_trials"],
                      "new_evaluation_trials": manifest["new_evaluation_trials"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--same-placement", action="store_true", required=True,
                        help="confirm electrodes have stayed on and unchanged since the reference recording")
    args = parser.parse_args()
    prepare(args.reference, args.output or Path("data/collection") / datetime.now().strftime("compact_%Y%m%d_%H%M%S"))
