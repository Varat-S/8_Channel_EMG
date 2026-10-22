from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import time
import unittest

from emg.types import Sample
from emg.vision_capture import Candidate, VISION_GESTURES, VisionCaptureController


class FastSource:
    unit = "test_count"

    def __init__(self) -> None:
        self.index = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def reconnect(self) -> None:
        return None

    def read_sample(self) -> Sample:
        time.sleep(.002)
        sample = Sample(self.index, tuple(float(self.index + channel) for channel in range(8)))
        self.index += 1
        return sample


class StallingSource:
    unit = "test_count"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def reconnect(self) -> None:
        time.sleep(.002)

    def read_sample(self) -> Sample:
        time.sleep(.002)
        raise TimeoutError("test stream pause")


class OnePauseSource(FastSource):
    def __init__(self) -> None:
        super().__init__()
        self.paused = False

    def read_sample(self) -> Sample:
        if self.index >= 5 and not self.paused:
            self.paused = True
            time.sleep(.01)
            raise TimeoutError("one transient pause")
        return super().read_sample()


def observation(label: str) -> dict:
    return {
        "label": label,
        "confidence": .95,
        "clientTimeMs": time.time() * 1000,
        "mediaTimeMs": 100,
        "handLandmarks": [{"x": 0, "y": 0, "z": 0}] * 21,
        "poseLandmarks": [],
        "metrics": {
            "armVisibility": .95,
            "elbowAngleDegrees": None,
            "elbowPositionX": .25,
            "elbowPositionY": .60,
            "forearmElevationDegrees": 25,
            "wristDeflectionDegrees": 2,
            "postureBucket": "shoulder_out_of_frame",
            "forearmOrientation": "horizontal",
        },
    }


def wait_for(predicate, seconds: float = 2.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("timed out waiting for capture state")


class VisionCaptureTests(unittest.TestCase):
    def test_transient_timeout_returns_phase_to_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            events = []
            controller = VisionCaptureController(
                Path(temporary), OnePauseSource, events.append,
                source_name="test", source_unit="test_count",
            )
            controller.start_session({
                "subject_id": "TEST", "session_id": "pause_test",
                "electrode_placement_id": "P3", "arm_posture": "reference",
                "vision_calibration": {
                    "pose_templates": {
                        label: {
                            "prototypes": [
                                {"posture": posture, "mean": [0.0] * 10, "std": [0.1] * 10, "sampleCount": 18}
                                for posture in ("vertical", "horizontal")
                            ]
                        }
                        for label in VISION_GESTURES
                    },
                },
            })
            wait_for(lambda: any(event.get("phase") == "reconnecting" for event in events))
            wait_for(lambda: any(
                event.get("message") == "EMG stream resumed. Continue with the next pose."
                for event in events
            ))
            self.assertEqual(controller.status()["phase"], "awaiting_baseline")
            controller.stop_session()

    def test_accepts_synchronized_baseline_and_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            events = []
            controller = VisionCaptureController(
                Path(temporary), FastSource, events.append,
                source_name="test", source_unit="test_count",
                baseline_seconds=.032, trial_seconds=.032,
                observation_timeout_seconds=2,
            )
            controller.start_session({
                "subject_id": "TEST", "session_id": "vision_test",
                "electrode_placement_id": "P3", "arm_posture": "reference",
                "vision_calibration": {
                    "pose_templates": {
                        label: {
                            "prototypes": [
                                {"posture": posture, "mean": [0.0] * 10, "std": [0.1] * 10, "sampleCount": 18}
                                for posture in ("vertical", "horizontal")
                            ]
                        }
                        for label in VISION_GESTURES
                    },
                },
            })
            wait_for(lambda: controller.status()["phase"] == "awaiting_baseline")
            controller.observe(observation("rest"))
            controller.start_baseline()
            controller.observe(observation("rest"))
            controller.observe(observation("rest"))
            wait_for(lambda: controller.status()["baselineComplete"])
            self.assertTrue(any(event.get("type") == "vision_progress" for event in events))
            controller.observe(observation("fist"))
            controller.start_trial("fist")
            controller.observe(observation("fist"))
            controller.observe(observation("fist"))
            wait_for(lambda: controller.status()["counts"]["fist"] == 1)
            controller.stop_session()

            directory = Path(temporary) / "TEST" / "vision_test"
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertTrue(metadata["session_complete"])
            self.assertEqual(metadata["accepted_counts"]["fist"], 1)
            self.assertEqual(len(metadata["vision_calibration"]["pose_templates"]), 7)
            self.assertEqual(len(metadata["trials"]), 1)
            with (directory / "samples.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            fist_rows = [row for row in rows if row["phase"] == "gesture"]
            self.assertEqual(len(fist_rows), 4)
            self.assertTrue(all(row["gesture"] == "fist" for row in fist_rows))
            self.assertTrue(all(row["vision_elbow_posture"] == "shoulder_out_of_frame" for row in fist_rows))
            self.assertTrue(all(row["vision_forearm_orientation"] == "horizontal" for row in fist_rows))
            self.assertTrue((directory / "vision_observations.jsonl").exists())

    def test_camera_dropout_allowance_is_based_on_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = VisionCaptureController(
                Path(temporary), FastSource, lambda _event: None,
                source_name="test", source_unit="test_count",
            )
            baseline = Candidate(
                "baseline", "rest", "baseline", 625, time.monotonic(), .95, {}
            )
            trial = Candidate(
                "trial", "fist", "trial", 312, time.monotonic(), .95, {}
            )
            baseline.observation_count = 20
            trial.observation_count = 10
            self.assertEqual(controller._allowed_invalid_observations(baseline), 7)
            self.assertEqual(controller._allowed_invalid_observations(trial), 4)

    def test_stop_saves_while_receiver_is_reconnecting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            events = []
            controller = VisionCaptureController(
                Path(temporary), StallingSource, events.append,
                source_name="test", source_unit="test_count",
            )
            controller.start_session({
                "subject_id": "TEST", "session_id": "stalled_test",
                "electrode_placement_id": "P3", "arm_posture": "reference",
                "vision_calibration": {
                    "pose_templates": {
                        label: {
                            "prototypes": [
                                {"posture": posture, "mean": [0.0] * 10, "std": [0.1] * 10, "sampleCount": 18}
                                for posture in ("vertical", "horizontal")
                            ]
                        }
                        for label in VISION_GESTURES
                    },
                },
            })
            wait_for(lambda: controller.status()["phase"] == "reconnecting")
            status = controller.stop_session()
            self.assertFalse(status["running"])
            self.assertEqual(status["phase"], "complete")
            metadata = json.loads(
                (Path(temporary) / "TEST" / "stalled_test" / "metadata.json").read_text()
            )
            self.assertTrue(metadata["session_complete"])
            self.assertTrue(any(event.get("phase") == "stopping" for event in events))


if __name__ == "__main__":
    unittest.main()
