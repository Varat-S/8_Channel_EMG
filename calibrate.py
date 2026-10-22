"""Run a prompted calibration session from synthetic, replayed, or live data."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import sys

# Allow this repository script to run before/without installing the local package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emg.calibration import CalibrationConfig, CalibrationRunner
from emg.recording import SessionRecorder
from emg.sources import SerialFrameSource, SyntheticSource, TextReplaySource
from emg.session_metadata import validate_session_conditions
from emg.dataset_v2 import (
    DatasetV2Protocol, DatasetV2Runner, DatasetV2SyntheticSource, FIVE_CORE_GESTURES, V2_GESTURES,
    V2_ROW_FIELDS, parse_v2_conditions, v2_metadata, validate_v2_recording,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("synthetic", "replay", "serial"), default="synthetic"
    )
    parser.add_argument("--replay-file", type=Path)
    parser.add_argument("--port", default="COM10")
    parser.add_argument("--baud", type=int, default=921_600)
    parser.add_argument("--protocol", choices=("legacy", "dataset_v2"), default="legacy")
    parser.add_argument("--subject", "--subject-id", dest="subject")
    parser.add_argument("--session", "--session-id", dest="session")
    parser.add_argument("--placement-id")
    parser.add_argument("--session-role", choices=("A", "B", "C"), default="A")
    parser.add_argument("--postures", nargs="+")
    parser.add_argument("--posture-chunk", action="store_true",
                        help="record selected posture trials; normalization and independent rest remain in reference")
    parser.add_argument("--randomize-postures", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plan", action="store_true", help="print Dataset v2 schedule without opening a source")
    parser.add_argument(
        "--posture",
        default="unspecified",
        help="short standardized posture identifier stored in session metadata",
    )
    parser.add_argument(
        "--gestures", nargs="+"
    )
    parser.add_argument("--trials", "--repetitions", dest="trials", type=int)
    parser.add_argument("--baseline-seconds", type=float)
    parser.add_argument("--preparation-seconds", type=float)
    parser.add_argument("--gesture-seconds", type=float)
    parser.add_argument("--rest-seconds", type=float)
    parser.add_argument("--independent-rest-seconds", type=float, default=60.0)
    parser.add_argument("--posture-change-seconds", type=float, default=15.0)
    parser.add_argument(
        "--fast", action="store_true", help="skip prompt countdowns for software tests"
    )
    parser.add_argument("--output-root", type=Path, default=Path("recordings"))
    parser.add_argument("--conditions-json", type=Path,
                        help="optional JSON object of recorded posture/electrode/date conditions")
    args = parser.parse_args()
    if args.source == "replay" and args.replay_file is None:
        parser.error("--replay-file is required when --source replay is selected")
    try:
        values = json.loads(args.conditions_json.read_text(encoding="utf-8")) if args.conditions_json else {}
        if args.protocol == "dataset_v2":
            args.conditions, args.block_conditions = parse_v2_conditions(values)
            for argument, field in (("session", "session_id"), ("subject", "subject_id"), ("placement_id", "electrode_placement_id")):
                supplied = getattr(args, argument)
                recorded = args.conditions.get(field)
                if supplied and recorded and supplied != recorded:
                    raise ValueError(f"--{argument.replace('_', '-')} conflicts with conditions {field}")
                setattr(args, argument, supplied or recorded)
            if not args.session or not args.placement_id:
                raise ValueError("Dataset v2 requires session and placement IDs (CLI or conditions JSON)")
            defaults = dict(trials=6, baseline_seconds=5.0, preparation_seconds=3.0, gesture_seconds=3.0, rest_seconds=2.0)
            preset = args.conditions.get("gesture_set")
            if args.gestures is None:
                args.gestures = list(FIVE_CORE_GESTURES if preset == "five_core_5class" else V2_GESTURES)
            if args.conditions.get("gesture_set") == "dataset_v2_9class" and set(args.gestures) != set(V2_GESTURES):
                raise ValueError("dataset_v2_9class conditions conflict with the custom --gestures; use a custom gesture_set name")
            if preset == "five_core_5class" and set(args.gestures) != set(FIVE_CORE_GESTURES):
                raise ValueError("five_core_5class conditions conflict with --gestures")
        else:
            args.conditions = validate_session_conditions(values)
            defaults = dict(trials=2, baseline_seconds=2.0, preparation_seconds=3.0, gesture_seconds=2.0, rest_seconds=1.0)
            args.gestures = args.gestures or ["fist", "open_hand", "pinch"]
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        args.subject = args.subject or "DRYRUN"
        if args.trials <= 0:
            raise ValueError("--trials must be positive")
    except (ValueError, OSError) as error:
        parser.error(f"invalid condition metadata: {error}")
    return args


def make_source(args: argparse.Namespace):
    if args.source == "synthetic":
        source = DatasetV2SyntheticSource(args.seed) if args.protocol == "dataset_v2" else SyntheticSource()
        unsupported = set(args.gestures) - set(source.supported_gestures)
        if unsupported:
            raise ValueError(f"synthetic source does not support: {sorted(unsupported)}")
        return nullcontext(source)
    if args.source == "replay":
        return TextReplaySource(args.replay_file)
    return SerialFrameSource(port=args.port, baud=args.baud)


def main() -> int:
    args = parse_args()
    if args.protocol == "dataset_v2":
        try:
            return run_dataset_v2(args)
        except (ValueError, OSError, KeyError, EOFError) as error:
            print(f"Dataset v2 recording error: {error}", file=sys.stderr)
            return 2
    session_id = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = args.output_root / args.subject / session_id
    config = CalibrationConfig(
        gestures=tuple(args.gestures),
        trials_per_gesture=args.trials,
        baseline_seconds=args.baseline_seconds,
        preparation_seconds=args.preparation_seconds,
        gesture_seconds=args.gesture_seconds,
        rest_seconds=args.rest_seconds,
    )

    with make_source(args) as source:
        metadata = {
            "subject_id": args.subject,
            "session_id": session_id,
            "posture": args.posture,
            "source": args.source,
            "value_unit": source.unit,
            "gestures": list(config.gestures),
            "trials_per_gesture": config.trials_per_gesture,
            "baseline_seconds": config.baseline_seconds,
            "preparation_seconds": config.preparation_seconds,
            "gesture_seconds": config.gesture_seconds,
            "rest_seconds": config.rest_seconds,
            "protocol_interpretation": (
                "38-byte provisional big-endian decoder" if args.source == "serial" else None
            ),
            "label_timing_valid": True,
            **args.conditions,
        }
        with SessionRecorder(directory, metadata) as recorder:
            print(f"Recording to {directory}", flush=True)
            CalibrationRunner(
                source,
                recorder,
                config,
                announce=lambda message: print(message, flush=True),
                wait_for_prompts=not args.fast,
            ).run()
            recorder.mark_complete()

    print("Review samples.csv, events.csv, and metadata.json before using the session.")
    return 0


def run_dataset_v2(args) -> int:
    protocol = DatasetV2Protocol(
        session_role=args.session_role, gestures=tuple(args.gestures),
        postures=tuple(args.postures) if args.postures else None, repetitions=args.trials, seed=args.seed,
        baseline_seconds=args.baseline_seconds, preparation_seconds=args.preparation_seconds,
        gesture_seconds=args.gesture_seconds, rest_seconds=args.rest_seconds,
        independent_rest_seconds=args.independent_rest_seconds, posture_change_seconds=args.posture_change_seconds,
        randomize_postures=args.randomize_postures, block_conditions=args.block_conditions,
        posture_chunk=args.posture_chunk)
    metadata = v2_metadata(args.session, args.subject, args.placement_id, protocol, args.conditions,
                           args.source, "pending")
    if args.plan:
        print(json.dumps(metadata, indent=2))
        return 0
    directory = args.output_root / args.subject / args.session
    if directory.exists():
        raise ValueError(f"session directory already exists: {directory}")
    with make_source(args) as source:
        metadata["value_unit"] = source.unit
        with SessionRecorder(directory, metadata, condition_fields=V2_ROW_FIELDS) as recorder:
            print(f"Recording Dataset v2 to {directory}. Role {args.session_role}; placement {args.placement_id}.", flush=True)
            DatasetV2Runner(source, recorder, protocol, announce=lambda message: print(message, flush=True),
                            wait_for_prompts=not args.fast).run()
            # Validate before marking complete; failed acquisitions remain incomplete.
            recorder.flush()
            validate_v2_recording(directory, require_complete=False)
            recorder.mark_complete()
    validation = validate_v2_recording(directory)
    print(json.dumps(validation, indent=2))
    print("Review raw signals before using this session for real-data analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
