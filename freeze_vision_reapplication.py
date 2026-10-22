"""Freeze the seven gesture model and a prospective reapplication protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


GESTURES = ("rest", "fist", "splayed", "peace", "point", "pinch", "thumbs_up")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(root: Path) -> dict:
    source = root / "recordings/S01/vision_20261016_192456"
    models = root / "models/vision_posture_20261016_192456"
    report = root / "results/vision_posture_20261016_192456/summary.json"
    files = [
        source / "samples.csv", source / "events.csv", source / "metadata.json",
        source / "protocol_posture.json", models / "lda_all_postures.joblib",
        models / "cnn_all_postures.pt", report,
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    return {
        "protocol": "seven_gesture_electrode_reapplication_v1",
        "source_session": "vision_20261016_192456",
        "source_files_sha256": {str(path.relative_to(root)): sha256(path) for path in files},
        "gestures": list(GESTURES),
        "models": {
            "primary": "models/vision_posture_20261016_192456/cnn_all_postures.pt",
            "reference": "models/vision_posture_20261016_192456/lda_all_postures.joblib",
        },
        "test_design": {
            "application": "new electrode application, same participant and nominal placement",
            "postures": ["vertical", "horizontal"],
            "baseline": "collect a fresh relaxed baseline in each posture before any gestures",
            "trials_per_gesture_per_posture": 6,
            "calibration_repetitions": [1, 2, 3],
            "fixed_test_repetitions": [4, 5, 6],
            "zero_shot": "score frozen models on fixed test repetitions before fitting or tuning",
            "adaptation_curve": "fit using nested calibration repetitions 1, then 1-2, then 1-3; always score the same fixed test repetitions 4-6",
            "primary_metric": "trial grouped smoothed macro F1 across seven gestures",
            "secondary_metrics": ["raw macro F1", "per gesture recall", "false active fraction on independent rest", "prediction delay"],
            "posture_truth": "operator supplied block order; keep camera orientation as an audit measurement",
            "no_test_feedback": "do not change model, threshold, trial exclusion or preprocessing after inspecting fixed test scores",
        },
        "normalization_comparison": [
            "strict zero shot uses frozen original baseline scale",
            "baseline adaptation uses only fresh rest baseline and keeps model weights frozen",
        ],
        "electrode_check": [
            "confirm transmitter battery and receiver stream",
            "collect a short rest and fist pilot; flag flat or saturated channels",
            "record placement photo or written landmarks and reapplication ID",
        ],
        "status": "frozen before new electrode data",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/vision_reapplication_protocol_v1.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen plan: {args.output}")
    plan = build_plan(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(plan['source_files_sha256'])} inputs in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
