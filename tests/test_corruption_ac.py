import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from emg.evaluation.corruptions import GaussianNoise, PowerlineInterference, ac_signal_power, sweep_conditions
from emg.evaluation.corruption_benchmark import run_corruption_benchmark, signal_scale_diagnostics
from emg.evaluation.dataset import (
    group_sessions_by_metadata, load_raw_recordings, select_sessions, write_manifest,
)
from emg.session_metadata import OPTIONAL_SESSION_FIELDS, validate_session_conditions
from test_benchmark import create_recording, run_args


class AcScalingTests(unittest.TestCase):
    def test_ac_rms_independent_of_huge_dc_offset(self):
        time = np.arange(1250) / 125
        signal = 20 * np.sin(2 * np.pi * 15 * time[:, None]) * np.arange(1, 9)
        power = ac_signal_power(signal)
        for offset in (1000, 1e6, 1e9):
            np.testing.assert_allclose(np.sqrt(ac_signal_power(signal + offset)), np.sqrt(power), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(np.sqrt(power), 20 * np.arange(1, 9) / np.sqrt(2), rtol=1e-12)

    def test_corrected_broadband_snr_uses_ac_power(self):
        samples = 1e6 + np.random.default_rng(2).normal(size=(50000, 8)) * np.arange(1, 9)
        for target in (30, 25, 20, 15, 10, 5, 0):
            noisy = GaussianNoise(target).apply(samples, 125, np.random.default_rng(7))
            power = np.mean((samples - samples.mean(axis=0)) ** 2, axis=0)
            measured = 10 * np.log10(power / np.mean((noisy - samples) ** 2, axis=0))
            np.testing.assert_allclose(measured, target, atol=0.12)

    def test_corrected_powerline_peak_amplitude_even_with_dc(self):
        time = np.arange(1250) / 125
        signal = 4 * np.sin(2 * np.pi * 15 * time[:, None]) * np.arange(1, 9)
        raw = signal + 1e6
        for multiplier in (0.01, 0.025, 0.05, 0.1, 0.25, 0.5):
            added = PowerlineInterference(multiplier).apply(raw, 125, np.random.default_rng(7)) - raw
            rms = np.sqrt(np.mean(added ** 2, axis=0))
            np.testing.assert_allclose(rms * np.sqrt(2), multiplier * np.sqrt(ac_signal_power(raw)), rtol=1e-8, atol=1e-9)
            spectrum = np.abs(np.fft.rfft(added, axis=0))
            frequency = np.fft.rfftfreq(len(raw), 1 / 125)
            np.testing.assert_array_equal(frequency[np.argmax(spectrum, axis=0)], np.full(8, 50.0))

    def test_constant_dc_channel_gets_no_ac_scaled_noise(self):
        raw = np.full((1000, 8), 1e6)
        for corruption in (GaussianNoise(0), PowerlineInterference(0.5)):
            np.testing.assert_array_equal(corruption.apply(raw, 125, np.random.default_rng(7)), raw)

    def test_single_channel_ac_noise_preserves_others_and_original(self):
        raw = 1e6 + np.random.default_rng(4).normal(size=(1000, 8))
        before = raw.copy()
        noisy = GaussianNoise(10, (3,)).apply(raw, 125, np.random.default_rng(7))
        np.testing.assert_array_equal(noisy[:, [0, 1, 2, 4, 5, 6, 7]], raw[:, [0, 1, 2, 4, 5, 6, 7]])
        self.assertFalse(np.array_equal(noisy[:, 3], raw[:, 3]))
        np.testing.assert_array_equal(raw, before)

    def test_current_sweep_finer_levels_and_custom_channel_snr(self):
        self.assertEqual(len(sweep_conditions()), 54)
        conditions = sweep_conditions("powerline")
        self.assertEqual(len(set(row.condition_id for row in conditions)), 7)
        self.assertEqual([row.severity for row in conditions[1:]], [0.01, 0.025, 0.05, 0.1, 0.25, 0.5])
        noise = sweep_conditions("channel-noise", channel_noise_snr=[20, 15, 10, 5, 0])
        self.assertEqual(len(noise), 41)
        self.assertTrue(all(row.corruption.power_reference == "ac" for row in noise[1:]))
        with self.assertRaises(ValueError):
            sweep_conditions(channel_noise_snr=[20, 20])


class FutureMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directory = create_recording(self.root, "a")
        self.manifest = self.root / "data/session_manifest.csv"
        write_manifest(self.root / "recordings", self.manifest)

    def tearDown(self):
        self.temporary.cleanup()

    def test_legacy_manifest_without_new_fields_still_loads(self):
        with self.manifest.open(newline="") as file:
            rows = list(csv.DictReader(file))
        fields = [key for key in rows[0] if key not in OPTIONAL_SESSION_FIELDS or key in ("recording_date", "notes")]
        with self.manifest.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        sessions, _ = select_sessions(self.manifest, ("rest", "fist"), ["a"])
        self.assertEqual(len(sessions), 1)
        for field in ("arm_posture", "elbow_position", "shoulder_position", "forearm_orientation", "electrode_reapplied", "dynamic_arm_motion"):
            self.assertIsNone(sessions[0].metadata[field])

    def test_explicit_conditions_filter_and_group_without_inference(self):
        second = create_recording(self.root, "b")
        for directory, posture, reapplied in ((self.directory, "reference", False), (second, "elbow_flexed", True)):
            path = directory / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata.update(arm_posture=posture, electrode_reapplied=reapplied, forearm_orientation="neutral",
                            elbow_position="90_deg_flexion", electrode_placement_id="placement_A", recording_date="2026-10-06")
            path.write_text(json.dumps(metadata))
        write_manifest(self.root / "recordings", self.manifest)
        sessions, excluded = select_sessions(self.manifest, ("rest", "fist"), ["a", "b"], {"electrode_reapplied": "true"})
        self.assertEqual([session.session_id for session in sessions], ["b"])
        self.assertIn("metadata filter", excluded[0]["reason"])
        all_sessions, _ = select_sessions(self.manifest, ("rest", "fist"), ["a", "b"])
        groups = group_sessions_by_metadata(all_sessions, ("arm_posture", "electrode_reapplied"))
        self.assertEqual(set(groups), {("reference", False), ("elbow_flexed", True)})
        date_groups = group_sessions_by_metadata(all_sessions, ("recording_date",))
        self.assertEqual(set(date_groups), {("2026-10-06",)})

    def test_manifest_annotations_can_supply_optional_unknown_values(self):
        with self.manifest.open(newline="") as file:
            rows = list(csv.DictReader(file))
        rows[0]["arm_posture"], rows[0]["dynamic_arm_motion"] = "custom", "false"
        with self.manifest.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        sessions, _ = select_sessions(self.manifest, ("rest", "fist"), ["a"], {"arm_posture": "custom", "dynamic_arm_motion": "false"})
        self.assertEqual(len(sessions), 1)
        self.assertIs(sessions[0].metadata["dynamic_arm_motion"], False)
        self.assertNotIn("arm_posture", json.loads((self.directory / "metadata.json").read_text()))

    def test_condition_validation(self):
        for values in ({"electrode_reapplied": "true"}, {"arm_posture": "made_up"},
                       {"forearm_orientation": "angle_99"}, {"session_id": "override"}, {"recording_date": "06/10/2026"}):
            with self.assertRaises(ValueError):
                validate_session_conditions(values)
        with self.assertRaisesRegex(ValueError, "unknown manifest filter"):
            select_sessions(self.manifest, ("rest", "fist"), ["a"], {"typo": "reference"})

    def test_calibration_cli_saves_conditions(self):
        project = Path(__file__).resolve().parents[1]
        conditions = self.root / "conditions.json"
        conditions.write_text(json.dumps({"arm_posture": "reference", "electrode_reapplied": False,
                                          "dynamic_arm_motion": False, "elbow_position": "90_deg_flexion"}))
        result = subprocess.run([sys.executable, str(project / "calibrate.py"), "--source", "synthetic", "--fast",
                                 "--subject", "DRYRUN", "--session", "conditions_test", "--gestures", "fist", "--trials", "1",
                                 "--baseline-seconds", "0.016", "--preparation-seconds", "0.016", "--gesture-seconds", "0.016", "--rest-seconds", "0.016",
                                 "--conditions-json", str(conditions), "--output-root", str(self.root / "new_recordings")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((self.root / "new_recordings/DRYRUN/conditions_test/metadata.json").read_text())
        self.assertEqual(metadata["arm_posture"], "reference")
        self.assertIs(metadata["electrode_reapplied"], False)
        self.assertTrue(metadata["session_complete"])

    def test_scale_diagnostics_and_ac_repeat_determinism(self):
        args = run_args(self.root, mode="corruption", output=self.root / "sweep")
        args.corruption, args.repeats, args.include_pair_dropout, args.seed = "channel-noise", 2, False, 7
        args.corruption_version, args.channel_noise_snr = "v1.1", [15]
        with contextlib.redirect_stdout(io.StringIO()):
            rows, config = run_corruption_benchmark(args)
        self.assertEqual(len(rows), 9)
        self.assertTrue((args.output / "signal_scale_summary.csv").is_file())
        before = (args.output / "predictions/channel_noise_CH1_snr15_seed7.csv.gz").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            repeated, _ = run_corruption_benchmark(args)
        self.assertEqual(rows, repeated)
        self.assertEqual(before, (args.output / "predictions/channel_noise_CH1_snr15_seed7.csv.gz").read_bytes())
        self.assertEqual(config["corruption_config"]["power_reference"], "ac")
        self.assertIn("including guards", config["injection_location"])


if __name__ == "__main__":
    unittest.main()
