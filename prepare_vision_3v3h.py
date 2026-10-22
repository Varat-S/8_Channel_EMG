"""Make an auditable 42-trial analysis copy from a mis-tagged reapplication run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


GESTURES = ("rest", "fist", "splayed", "peace", "point", "pinch", "thumbs_up")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("session_complete") or not metadata.get("source", "").startswith("serial:"):
        raise ValueError("expected a completed serial recording")
    if metadata.get("training_ready"):
        raise ValueError("source is already marked training-ready; no correction needed")
    if len(metadata.get("baselines", [])) != 1 or metadata["baselines"][0].get("sample_count") != 625:
        raise ValueError("expected one five-second baseline")
    trials = metadata.get("trials", [])
    observed = [(trial.get("gesture"), trial.get("repetition_index")) for trial in trials]
    expected = [(gesture, repetition) for gesture in GESTURES for repetition in range(1, 7)]
    if len(observed) != 42 or sorted(observed) != sorted(expected):
        raise ValueError("expected exactly six accepted trials for each of seven gestures")
    if any(trial.get("sample_count") != 312 for trial in trials):
        raise ValueError("unexpected trial length")
    if any(trial.get("protocol_posture") != "vertical" for trial in trials):
        raise ValueError("this correction applies only to the all-vertical recorder tag")
    source_hashes = {name: digest(source / name)
                     for name in ("samples.csv", "events.csv", "metadata.json", "vision_observations.jsonl")}
    destination.mkdir(parents=True)
    for name in ("samples.csv", "events.csv", "vision_observations.jsonl"):
        shutil.copy2(source / name, destination / name)
    metadata["training_ready"] = True
    metadata["analysis_override"] = {
        "purpose": "42-trial within-session study and deployment training",
        "original_training_ready": False,
        "original_protocol": metadata.get("protocol"),
        "source_metadata_sha256": source_hashes["metadata.json"],
        "reason": "participant reports repetitions 1-3 vertical and 4-6 horizontal; recorder posture block was not advanced",
        "not_the_frozen_84_trial_reapplication_protocol": True,
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    posture = {
        "format_version": 1,
        "session_id": metadata["session_id"],
        "label_source": "participant-reported fixed acquisition order",
        "authoritative_for_training": True,
        "applies_to_phase": "gesture",
        "applies_to_gestures": list(GESTURES),
        "repetition_rule": {"1-3": "vertical", "4-6": "horizontal"},
        "measured_field_preserved": "trials[].vision_posture.forearmOrientation",
        "notes": "The original recording and camera measurements remain unchanged. This analysis copy uses the participant's reported posture order.",
    }
    (destination / "protocol_posture.json").write_text(json.dumps(posture, indent=2) + "\n", encoding="utf-8")
    (destination / "source_manifest.json").write_text(json.dumps({
        "source": str(source.resolve()), "source_files_sha256": source_hashes,
        "source_protocol_status": "incomplete 84-trial protocol; 42-trial analysis copy only",
    }, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
