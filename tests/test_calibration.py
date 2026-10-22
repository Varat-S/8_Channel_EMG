import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emg.calibration import CalibrationConfig, CalibrationRunner
from emg.recording import SessionRecorder
from emg.sources import SyntheticSource


class CalibrationTests(unittest.TestCase):
    def test_fast_synthetic_session_has_expected_labels_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory) / "session"
            config = CalibrationConfig(
                gestures=("fist", "pinch"),
                trials_per_gesture=2,
                baseline_seconds=0.016,
                gesture_seconds=0.016,
                rest_seconds=0.016,
            )
            source = SyntheticSource(seed=1)
            with SessionRecorder(
                session, {"source": "synthetic", "value_unit": source.unit}
            ) as recorder:
                CalibrationRunner(
                    source,
                    recorder,
                    config,
                    announce=lambda _: None,
                    wait_for_prompts=False,
                ).run()

            with (session / "samples.csv").open(newline="", encoding="utf-8") as file:
                samples = list(csv.DictReader(file))
            with (session / "events.csv").open(newline="", encoding="utf-8") as file:
                events = list(csv.DictReader(file))
            metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))

            # 2 baseline samples plus four trials of 2 gesture + 2 rest samples.
            self.assertEqual(len(samples), 18)
            self.assertEqual(samples[0]["phase"], "baseline")
            self.assertEqual(samples[0]["gesture"], "rest")
            self.assertEqual(
                [row["gesture"] for row in samples if row["phase"] == "gesture"],
                ["fist", "fist", "pinch", "pinch"] * 2,
            )
            self.assertEqual(events[-1]["event"], "session_end")
            self.assertEqual(metadata["sample_rate_hz"], 125)
            self.assertEqual(metadata["value_unit"], "synthetic_count")

    def test_preparation_is_recorded_while_source_is_drained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory) / "session"
            config = CalibrationConfig(
                gestures=("fist",),
                trials_per_gesture=1,
                baseline_seconds=0.016,
                preparation_seconds=0.024,
                gesture_seconds=0.016,
                rest_seconds=0.016,
            )
            source = SyntheticSource(seed=2)
            with SessionRecorder(
                session, {"source": "synthetic", "value_unit": source.unit}
            ) as recorder:
                CalibrationRunner(
                    source,
                    recorder,
                    config,
                    announce=lambda _: None,
                    wait_for_prompts=True,
                ).run()

            with (session / "samples.csv").open(newline="", encoding="utf-8") as file:
                samples = list(csv.DictReader(file))
            self.assertEqual(
                [row["phase"] for row in samples],
                ["baseline"] * 2
                + ["prepare"] * 3
                + ["gesture"] * 2
                + ["rest"] * 2,
            )


if __name__ == "__main__":
    unittest.main()
