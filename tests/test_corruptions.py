import contextlib
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from emg.evaluation.benchmark import evaluate_model, prepare_benchmark
from emg.evaluation.corruptions import (
    ChannelDropout, GaussianNoise, NoCorruption, PowerlineInterference, sweep_conditions,
)
from emg.evaluation.corruption_benchmark import run_corruption_benchmark
from emg.evaluation.dataset import (
    corrupt_heldout_dataset, load_raw_recordings, segment_rng, write_manifest,
)
from test_benchmark import create_recording, run_args


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        self.samples = np.random.default_rng(4).normal(size=(2000, 8)) * np.arange(1, 9)

    def test_broadband_per_channel_target_snr(self):
        samples = np.random.default_rng(2).normal(size=(50000, 8)) * np.arange(1, 9)
        for target in (20, 10, 5, 0):
            noisy = GaussianNoise(target, power_reference="raw").apply(samples, 125, np.random.default_rng(7))
            measured = 10 * np.log10(np.mean(samples ** 2, axis=0) / np.mean((noisy - samples) ** 2, axis=0))
            np.testing.assert_allclose(measured, target, atol=0.12)

    def test_powerline_frequency_peak_amplitude_and_independent_phases(self):
        samples = np.ones((1250, 8)) * np.arange(1, 9)
        result = PowerlineInterference(0.5, power_reference="raw").apply(samples, 125, np.random.default_rng(7))
        interference = result - samples
        spectrum = np.abs(np.fft.rfft(interference, axis=0))
        frequencies = np.fft.rfftfreq(len(samples), 1 / 125)
        np.testing.assert_array_equal(frequencies[np.argmax(spectrum, axis=0)], np.full(8, 50.0))
        np.testing.assert_allclose(np.sqrt(np.mean(interference ** 2, axis=0)),
                                   0.5 * np.arange(1, 9) / np.sqrt(2), atol=1e-12)
        self.assertFalse(np.allclose(interference[:, 0], interference[:, 1] / 2))

    def test_dropout_only_selected_channels_and_pair_support(self):
        for channels in ((3,), (1, 6)):
            result = ChannelDropout(channels).apply(self.samples, 125, np.random.default_rng(7))
            np.testing.assert_array_equal(result[:, channels], 0)
            remaining = [ch for ch in range(8) if ch not in channels]
            np.testing.assert_array_equal(result[:, remaining], self.samples[:, remaining])

    def test_single_channel_noise_only_selected_channel(self):
        result = GaussianNoise(0, (4,), name="single_channel_noise").apply(self.samples, 125, np.random.default_rng(7))
        self.assertFalse(np.array_equal(result[:, 4], self.samples[:, 4]))
        remaining = [ch for ch in range(8) if ch != 4]
        np.testing.assert_array_equal(result[:, remaining], self.samples[:, remaining])

    def test_all_corruptions_deterministic_and_never_mutate_inputs(self):
        before = self.samples.copy()
        for corruption in (NoCorruption(), GaussianNoise(10), PowerlineInterference(0.25), ChannelDropout((1,)), GaussianNoise(0, (2,))):
            a = corruption.apply(self.samples, 125, np.random.default_rng(7))
            b = corruption.apply(self.samples, 125, np.random.default_rng(7))
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(self.samples, before)
            self.assertFalse(np.shares_memory(a, self.samples))

    def test_zero_and_near_zero_power_finite_without_noise_floor(self):
        samples = np.zeros((100, 8))
        samples[:, 1] = 1e-200  # Mean-square underflows safely to zero.
        samples[:, 2] = 1e-100
        for corruption in (GaussianNoise(0, power_reference="raw"), PowerlineInterference(1, power_reference="raw")):
            result = corruption.apply(samples, 125, np.random.default_rng(7))
            self.assertTrue(np.all(np.isfinite(result)))
            np.testing.assert_array_equal(result[:, 0], 0)
            np.testing.assert_array_equal(result[:, 1], samples[:, 1])
            self.assertFalse(np.array_equal(result[:, 2], samples[:, 2]))

    def test_power_definition_includes_constant_dc(self):
        samples = np.full((1000, 8), 100.0)
        noisy = GaussianNoise(0, power_reference="raw").apply(samples, 125, np.random.default_rng(7))
        self.assertGreater(np.sqrt(np.mean((noisy - samples) ** 2)), 90)

    def test_invalid_configs_rejected(self):
        for channels in ((), (8,), (-1,), (2, 2), (1.5,)):
            with self.assertRaises(ValueError):
                ChannelDropout(channels).apply(self.samples, 125, np.random.default_rng(7))
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            PowerlineInterference(0.1).apply(self.samples, 100, np.random.default_rng(7))
        with self.assertRaises(ValueError):
            GaussianNoise(float("nan")).apply(self.samples, 125, np.random.default_rng(7))
        with self.assertRaises(ValueError):
            NoCorruption().apply(np.ones((10, 7)), 125, np.random.default_rng(7))

    def test_sweep_counts_and_optional_pairs(self):
        self.assertEqual(len(sweep_conditions(version="v1")), 25)
        self.assertEqual(len(sweep_conditions(include_pair_dropout=True, version="v1")), 53)
        self.assertEqual(len(sweep_conditions("channel-dropout", True)), 37)
        self.assertEqual(len(sweep_conditions("none")), 1)

    def test_segment_substreams_reset_stably(self):
        a = segment_rng(7, "session", 1).normal(size=50)
        np.testing.assert_array_equal(a, segment_rng(7, "session", 1).normal(size=50))
        self.assertFalse(np.array_equal(a, segment_rng(7, "session", 2).normal(size=50)))
        self.assertFalse(np.array_equal(a, segment_rng(7, "other-session", 1).normal(size=50)))


class CorruptionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        create_recording(self.root, "a")
        write_manifest(self.root / "recordings", self.root / "data/session_manifest.csv")
        self.prepared = prepare_benchmark(run_args(self.root))
        self.recordings = load_raw_recordings(self.prepared.sessions)

    def tearDown(self):
        self.temporary.cleanup()

    def test_no_corruption_overlay_exactly_matches_v0(self):
        dataset, _ = corrupt_heldout_dataset(self.prepared.dataset, self.recordings, self.prepared.test_indices,
                                           self.prepared.run_config["pipeline"], NoCorruption(), 7)
        np.testing.assert_array_equal(dataset.data.features, self.prepared.dataset.data.features)
        result = evaluate_model(self.prepared.model, dataset, self.prepared.test_indices, self.prepared.run_config["pipeline"])
        self.assertEqual(result.metrics, self.prepared.result.metrics)
        np.testing.assert_array_equal(result.smoothed_probabilities, self.prepared.result.smoothed_probabilities)

    def test_corruption_preserves_model_training_features_raw_inputs_and_split(self):
        before = pickle.dumps(self.prepared.model, protocol=5)
        raw_copy = self.recordings[0].samples.copy()
        source_path = self.recordings[0].session.path / "samples.csv"
        file_before = source_path.read_bytes()
        config = self.prepared.run_config["pipeline"]
        results = []
        for _ in range(2):
            dataset, audit = corrupt_heldout_dataset(self.prepared.dataset, self.recordings, self.prepared.test_indices,
                                                   config, GaussianNoise(0), 7)
            np.testing.assert_array_equal(dataset.data.features[self.prepared.train_indices],
                                          self.prepared.dataset.data.features[self.prepared.train_indices])
            self.assertFalse(np.array_equal(dataset.data.features[self.prepared.test_indices],
                                           self.prepared.dataset.data.features[self.prepared.test_indices]))
            self.assertEqual({row["group_id"] for row in audit}, set(self.prepared.dataset.data.groups[self.prepared.test_indices]))
            result = evaluate_model(self.prepared.model, dataset, self.prepared.test_indices, config)
            results.append(result)
        self.assertEqual(results[0].metrics, results[1].metrics)
        np.testing.assert_array_equal(results[0].raw_probabilities, results[1].raw_probabilities)
        self.assertEqual(before, pickle.dumps(self.prepared.model, protocol=5))
        np.testing.assert_array_equal(raw_copy, self.recordings[0].samples)
        self.assertEqual(file_before, source_path.read_bytes())

    def test_raw_corruption_is_applied_only_to_full_heldout_segments(self):
        class Recorder:
            def __init__(self):
                self.samples = []

            def apply(self, samples, sampling_rate, rng):
                self.samples.append(samples.copy())
                return samples.copy()

        recorder = Recorder()
        _, audit = corrupt_heldout_dataset(self.prepared.dataset, self.recordings, self.prepared.test_indices,
                                          self.prepared.run_config["pipeline"], recorder, 7)
        self.assertEqual(len(recorder.samples), len(set(self.prepared.dataset.data.groups[self.prepared.test_indices])))
        self.assertTrue(all(len(samples) == 125 for samples in recorder.samples))  # Includes both 38-sample guards.
        for actual, row in zip(recorder.samples, audit):
            np.testing.assert_array_equal(actual, self.recordings[0].samples[row["start_sample"]:row["end_sample"]])

    def test_sweep_repeats_determinism_and_frozen_split_seed(self):
        args = run_args(self.root, mode="corruption", output=self.root / "sweep")
        args.corruption, args.repeats, args.include_pair_dropout, args.seed = "broadband", 2, False, 19
        args.corruption_version = "v1"
        with contextlib.redirect_stdout(io.StringIO()):
            rows, config = run_corruption_benchmark(args)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["smoothed_macro_f1"], self.prepared.result.metrics["smoothed"]["macro_f1"])
        self.assertEqual(config["clean_split_seed"], 7)
        self.assertEqual(config["train_window_indices"], self.prepared.train_indices.tolist())
        self.assertEqual(rows[1]["seeds"], "19|20")
        self.assertEqual(rows[1]["repeats"], 2)
        self.assertGreaterEqual(rows[1]["smoothed_macro_f1_std"], 0)
        prediction_bytes = {path.name: path.read_bytes() for path in (args.output / "predictions").iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):
            repeated, _ = run_corruption_benchmark(args)
        self.assertEqual(rows, repeated)
        self.assertEqual(prediction_bytes, {path.name: path.read_bytes() for path in (args.output / "predictions").iterdir()})
        self.assertTrue((args.output / "plots/broadband_snr_curve.png").is_file())
        self.assertTrue((args.output / "recall_changes_smoothed.csv").is_file())
        clean_f1 = rows[0]["smoothed_macro_f1"]
        self.assertAlmostEqual(rows[1]["delta_smoothed_macro_f1"], rows[1]["smoothed_macro_f1"] - clean_f1)
        metrics = [json.loads((args.output / f"broadband_noise/broadband_snr20_seed{seed}/metrics.json").read_text())
                   for seed in (19, 20)]
        expected = np.std([run["smoothed"]["macro_f1"] for run in metrics], ddof=1)
        self.assertAlmostEqual(rows[1]["smoothed_macro_f1_std"], expected)

    def test_real_current_label_baseline_stored_precision_regression(self):
        project = Path(__file__).resolve().parents[1]
        manifest = project / "data/session_manifest.csv"
        metrics_path = project / "results/baseline_v0/metrics.json"
        if not manifest.is_file() or not metrics_path.is_file():
            self.skipTest("real recording and frozen baseline artifacts are not present")
        args = run_args(self.root, manifest=manifest, sessions=["web_20260926_192402"], gesture_set=None)
        with contextlib.redirect_stdout(io.StringIO()):
            prepared = prepare_benchmark(args)
        expected = json.loads(metrics_path.read_text())
        self.assertEqual(prepared.result.metrics, expected)
        dataset, audit = corrupt_heldout_dataset(prepared.dataset, load_raw_recordings(prepared.sessions),
                                               prepared.test_indices, prepared.run_config["pipeline"], NoCorruption(), 7)
        actual = evaluate_model(prepared.model, dataset, prepared.test_indices, prepared.run_config["pipeline"])
        self.assertEqual(actual.metrics, expected)
        self.assertAlmostEqual(actual.metrics["smoothed"]["macro_f1"], 0.8097042606516291, places=14)
        self.assertTrue(any(row["phase"] == "baseline" for row in audit))


if __name__ == "__main__":
    unittest.main()
