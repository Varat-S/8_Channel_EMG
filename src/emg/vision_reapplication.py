"""Frozen seven-gesture electrode reapplication schedule and provenance checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROTOCOL_NAME = "seven_gesture_electrode_reapplication_v1"
PROTOCOL_SHA256 = "53dd8b1b477f4c4775c89fe446bdf8f9a09721268fa5a855f7bd8132b906e4ec"
GESTURES = ("rest", "fist", "splayed", "peace", "point", "pinch", "thumbs_up")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path, root: Path, *, verify_hashes: bool = True) -> dict:
    path = Path(path)
    plan_hash = sha256(path)
    if plan_hash != PROTOCOL_SHA256:
        raise ValueError("frozen reapplication protocol file hash mismatch")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    design = protocol["test_design"]
    if (protocol.get("protocol") != PROTOCOL_NAME
            or tuple(protocol.get("gestures", [])) != GESTURES
            or design.get("postures") != ["vertical", "horizontal"]
            or design.get("trials_per_gesture_per_posture") != 6
            or design.get("calibration_repetitions") != [1, 2, 3]
            or design.get("fixed_test_repetitions") != [4, 5, 6]):
        raise ValueError("unsupported reapplication protocol")
    for relative, expected in protocol["source_files_sha256"].items():
        source = (Path(root) / relative.replace("\\", "/")).resolve()
        if not source.is_relative_to(Path(root).resolve()):
            raise ValueError(f"source path leaves project root: {relative}")
        if verify_hashes and sha256(source) != expected:
            raise ValueError(f"frozen source hash mismatch: {relative}")
    protocol["_sha256"] = plan_hash
    return protocol


def empty_counts(protocol: dict) -> dict[str, dict[str, int]]:
    return {posture: {label: 0 for label in protocol["gestures"]}
            for posture in protocol["test_design"]["postures"]}


def schedule(protocol: dict) -> list[dict]:
    """Operator block order and nested calibration/test assignment."""
    result = []
    for posture in protocol["test_design"]["postures"]:
        result.append({"posture": posture, "kind": "baseline", "gesture": "rest"})
        for repetition in range(1, 7):
            for gesture in protocol["gestures"]:
                result.append({"posture": posture, "kind": "trial", "gesture": gesture,
                               "repetition_index": repetition,
                               "split": "calibration" if repetition <= 3 else "test"})
    return result


def validate_session(metadata: dict, protocol: dict, *, allow_synthetic: bool = False) -> None:
    design = protocol["test_design"]
    if metadata.get("protocol") != PROTOCOL_NAME:
        raise ValueError("session protocol mismatch")
    if metadata.get("reapplication_plan_sha256") != protocol["_sha256"]:
        raise ValueError("session was not recorded against this frozen protocol")
    if not metadata.get("session_complete") or not metadata.get("training_ready"):
        raise ValueError("session is incomplete or not analysis-ready")
    if (metadata.get("sample_rate_hz") != 125
            or metadata.get("channel_count") != 8
            or not metadata.get("label_timing_valid")):
        raise ValueError("session rate, channel count, or label timing is invalid")
    if not allow_synthetic and not metadata.get("source", "").startswith("serial:"):
        raise ValueError("synthetic or non-serial session cannot be scored as electrode reapplication")
    baselines = metadata.get("baselines", [])
    if [item.get("protocol_posture") for item in baselines] != design["postures"]:
        raise ValueError("one accepted baseline is required before each posture block")
    expected = {(posture, gesture, repetition)
                for posture in design["postures"]
                for gesture in protocol["gestures"] for repetition in range(1, 7)}
    actual = [(trial.get("protocol_posture"), trial.get("gesture"), trial.get("repetition_index"))
              for trial in metadata.get("trials", [])]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("expected exactly 84 unique accepted posture/gesture/repetition trials")
    identities = [trial.get("trial_id") for trial in metadata["trials"]]
    if len(set(identities)) != 84 or set(identities) & {item.get("trial_id") for item in baselines}:
        raise ValueError("trial identities must be unique and distinct from baselines")
    by_posture = {posture: [] for posture in design["postures"]}
    for trial in metadata["trials"]:
        by_posture[trial["protocol_posture"]].append(trial)
    for posture, baseline in zip(design["postures"], baselines):
        if baseline["sample_count"] < 125 or baseline["end_sample"] <= baseline["start_sample"]:
            raise ValueError(f"invalid {posture} baseline")
        trials = by_posture[posture]
        if any(trial["start_sample"] < baseline["end_sample"] for trial in trials):
            raise ValueError(f"{posture} trial precedes its baseline")
        if any(trial["sample_count"] < 125 for trial in trials):
            raise ValueError(f"short trial in {posture} block")
        if not allow_synthetic and (baseline["sample_count"] != 625
                                    or any(trial["sample_count"] != 312 for trial in trials)):
            raise ValueError(f"unexpected frozen capture duration in {posture} block")
    if max(trial["end_sample"] for trial in by_posture["vertical"]) > baselines[1]["start_sample"]:
        raise ValueError("posture blocks overlap or appear out of order")
