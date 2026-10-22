from __future__ import annotations

import json
import csv
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from emg.recording import SessionRecorder
from emg.types import Sample
from emg.vision_reapplication import load_protocol, schedule, validate_session
from emg.vision_capture import VisionCaptureController
from evaluate_vision_reapplication import adaptation_curve, evaluate, main as evaluate_main


class VisionReapplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(ROOT / "data/vision_reapplication_protocol_v1.json", ROOT)

    def test_frozen_schedule_has_two_baselines_and_84_unique_trials(self):
        steps = schedule(self.protocol)
        self.assertEqual(len(steps), 86)
        self.assertEqual([step["posture"] for step in steps if step["kind"] == "baseline"],
                         ["vertical", "horizontal"])
        tests = [step for step in steps if step.get("split") == "test"]
        self.assertEqual(len(tests), 42)
        self.assertEqual({step["repetition_index"] for step in tests}, {4, 5, 6})

    def test_incomplete_session_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_session({"protocol": self.protocol["protocol"],
                              "reapplication_plan_sha256": self.protocol["_sha256"],
                              "session_complete": False}, self.protocol)

    def test_modified_frozen_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "protocol.json"
            text = (ROOT / "data/vision_reapplication_protocol_v1.json").read_text(encoding="utf-8")
            copy.write_text(text.replace('"status": "frozen before new electrode data"',
                                         '"status": "edited"'), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                load_protocol(copy, ROOT)

    def test_recorder_requires_completed_vertical_block_before_advance(self):
        class SlowSource:
            def __init__(self):
                self.index = 0
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def read_sample(self):
                time.sleep(.002)
                sample = Sample(self.index, (1.0,) * 8)
                self.index += 1
                return sample

        templates = {gesture: {"prototypes": [
            {"posture": posture, "mean": [0.0] * 10, "std": [0.1] * 10}
            for posture in ("vertical", "horizontal")]}
            for gesture in self.protocol["gestures"]}
        with tempfile.TemporaryDirectory() as temporary:
            controller = VisionCaptureController(Path(temporary), SlowSource, lambda _: None,
                source_name="test", source_unit="test_count",
                reapplication_protocol=self.protocol)
            controller.start_session({"subject_id": "TEST", "session_id": "reapply_test",
                "electrode_placement_id": "P4", "vision_calibration": {"pose_templates": templates}})
            deadline = time.monotonic() + 2
            while controller.status()["phase"] != "awaiting_baseline" and time.monotonic() < deadline:
                time.sleep(.005)
            try:
                self.assertEqual(controller.status()["protocolPosture"], "vertical")
                with self.assertRaisesRegex(ValueError, "complete the vertical"):
                    controller.advance_posture()
                with controller._lock:
                    controller._baseline_complete = True
                    for gesture in self.protocol["gestures"]:
                        controller._counts[gesture] = 6
                controller.advance_posture()
                self.assertEqual(controller.status()["protocolPosture"], "horizontal")
                self.assertFalse(controller.status()["baselineComplete"])
                with self.assertRaisesRegex(ValueError, "baseline"):
                    controller.start_trial("fist")
            finally:
                controller.stop_session()
            metadata = json.loads((Path(temporary) / "TEST/reapply_test/metadata.json").read_text())
            self.assertEqual(metadata["reapplication_plan_sha256"], self.protocol["_sha256"])
            self.assertFalse(metadata["training_ready"])

    def test_accepted_capture_records_operator_posture(self):
        class SlowSource:
            def __init__(self):
                self.index = 0
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def read_sample(self):
                time.sleep(.004)
                sample = Sample(self.index, tuple(float(self.index + channel) for channel in range(8)))
                self.index += 1
                return sample

        templates = {gesture: {"prototypes": [
            {"posture": posture, "mean": [0.0] * 10, "std": [0.1] * 10}
            for posture in ("vertical", "horizontal")]}
            for gesture in self.protocol["gestures"]}
        def observation(label):
            return {"label": label, "confidence": .95, "clientTimeMs": 1,
                    "mediaTimeMs": 1, "handLandmarks": [{"x": 0, "y": 0, "z": 0}] * 21,
                    "metrics": {"armVisibility": .95, "forearmOrientation": "vertical"}}
        def wait_for(predicate):
            deadline = time.monotonic() + 3
            while not predicate() and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertTrue(predicate())
        with tempfile.TemporaryDirectory() as temporary:
            controller = VisionCaptureController(Path(temporary), SlowSource, lambda _: None,
                source_name="test", source_unit="test_count", reapplication_protocol=self.protocol,
                baseline_seconds=.2, trial_seconds=.2, observation_timeout_seconds=2)
            controller.start_session({"subject_id": "TEST", "session_id": "one_trial",
                "electrode_placement_id": "P4", "vision_calibration": {"pose_templates": templates}})
            try:
                wait_for(lambda: controller.status()["phase"] == "awaiting_baseline")
                controller.observe(observation("rest"))
                controller.start_baseline()
                controller.observe(observation("rest"))
                controller.observe(observation("rest"))
                wait_for(lambda: controller.status()["baselineComplete"])
                controller.observe(observation("fist"))
                controller.start_trial("fist")
                controller.observe(observation("fist"))
                controller.observe(observation("fist"))
                wait_for(lambda: controller.status()["counts"]["fist"] == 1)
            finally:
                controller.stop_session()
            directory = Path(temporary) / "TEST/one_trial"
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(metadata["baselines"][0]["protocol_posture"], "vertical")
            self.assertEqual(metadata["trials"][0]["protocol_posture"], "vertical")
            self.assertEqual(metadata["trials"][0]["repetition_index"], 1)
            with (directory / "samples.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual({row["protocol_posture"] for row in rows if row["phase"] == "gesture"}, {"vertical"})

    def test_full_synthetic_session_evaluates_without_test_leakage(self):
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session"
            metadata = {"protocol": self.protocol["protocol"],
                        "reapplication_plan_sha256": self.protocol["_sha256"],
                        "source": "paced_synthetic", "session_complete": False,
                        "training_ready": True, "label_timing_valid": True,
                        "baselines": [], "trials": []}
            index = 0
            with SessionRecorder(session, metadata, condition_fields=("protocol_posture",)) as recorder:
                for posture in self.protocol["test_design"]["postures"]:
                    baseline_id = f"baseline_{posture}"
                    recorder.conditions = {"protocol_posture": posture}
                    start = index
                    for _ in range(125):
                        values = rng.normal(0, 100, size=8)
                        recorder.write_sample(Sample(index, tuple(values)), phase="baseline",
                                              gesture="rest", trial=baseline_id)
                        index += 1
                    recorder._metadata["baselines"].append({"trial_id": baseline_id,
                        "protocol_posture": posture, "start_sample": start,
                        "end_sample": index, "sample_count": index - start})
                    for repetition in range(1, 7):
                        for gesture_index, gesture in enumerate(self.protocol["gestures"]):
                            identity = f"{posture}_{gesture}_r{repetition}"
                            start = index
                            for _ in range(125):
                                values = rng.normal(0, 100, size=8)
                                values[gesture_index] += rng.normal(0, 80)
                                recorder.write_sample(Sample(index, tuple(values)), phase="gesture",
                                                      gesture=gesture, trial=identity)
                                index += 1
                            recorder._metadata["trials"].append({"trial_id": identity,
                                "protocol_posture": posture, "gesture": gesture,
                                "repetition_index": repetition, "start_sample": start,
                                "end_sample": index, "sample_count": index - start})
                recorder.update_metadata({})
                recorder.mark_complete()
            data, model, artifact, labels, test, calibration, fresh, zero_shot = evaluate(
                session, self.protocol, allow_synthetic=True)
            self.assertEqual(len(zero_shot), 4)
            self.assertEqual(len(set(data["groups"][test])), 42)
            self.assertEqual(len(set(data["groups"][calibration])), 42)
            self.assertFalse(set(data["groups"][test]) & set(data["groups"][calibration]))
            curve = adaptation_curve(data, model, artifact, labels, test, calibration, fresh)
            self.assertEqual([row["training_trial_count"] for row in curve], [14, 28, 42])
            output = Path(temporary) / "result"
            with patch.object(sys, "argv", ["evaluate_vision_reapplication.py", "--session", str(session),
                                           "--output", str(output), "--allow-synthetic"]):
                self.assertEqual(evaluate_main(), 0)
            self.assertTrue((output / "zero_shot.json").exists())
            self.assertEqual(len(json.loads((output / "summary.json").read_text())["calibration_curve"]), 3)
            with self.assertRaisesRegex(ValueError, "synthetic"):
                validate_session(json.loads((session / "metadata.json").read_text()), self.protocol)


if __name__ == "__main__":
    unittest.main()
