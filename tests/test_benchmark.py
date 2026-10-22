import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from emg.dataset import FeatureDataset, load_feature_dataset
from emg.evaluation.benchmark import (
    DEFAULT_CONFIG, evaluate_model, grouped_split, run_benchmark, validate_split,
)
from emg.evaluation.dataset import (
    BenchmarkDataset, WindowInfo, load_benchmark_dataset, select_sessions, write_manifest,
)
from emg.evaluation.metrics import calculate_metrics
from emg.model import save_model


class MetricTests(unittest.TestCase):
    def test_known_classification_metrics_and_episodes(self):
        true = ["rest"] * 6
        predicted = ["rest", "fist", "fist", "rest", "wrist_up", "rest"]
        metrics = calculate_metrics(true, predicted, ["a"] * 6, ["rest", "fist", "wrist_up"], 0.04)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 9)
        self.assertEqual(metrics["confusion_matrix"], [[3, 2, 1], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(metrics["false_active_windows"], 3)
        self.assertEqual(metrics["false_activation_episodes"], 2)
        self.assertEqual(metrics["false_active_window_fraction"], 0.5)
        self.assertAlmostEqual(metrics["false_activation_episodes_per_minute"], 500)
        self.assertEqual(metrics["switch_count"], 4)
        self.assertAlmostEqual(metrics["switches_per_second"], 20)

    def test_boundaries_are_not_switches_and_initial_activity_counts_once(self):
        metrics = calculate_metrics(["rest"] * 4, ["rest", "rest", "fist", "fist"],
                                    ["a", "a", "b", "b"], ["rest", "fist"], 0.04)
        self.assertEqual(metrics["switch_count"], 0)
        self.assertEqual(metrics["false_activation_episodes"], 1)
        self.assertAlmostEqual(metrics["switch_observation_seconds"], 0.08)

    def test_changing_active_labels_does_not_add_rest_episode(self):
        metrics = calculate_metrics(["rest"] * 3, ["fist", "wrist_up", "fist"],
                                    ["a"] * 3, ["rest", "fist", "wrist_up"], 0.04)
        self.assertEqual(metrics["false_activation_episodes"], 1)

    def test_multiclass_values(self):
        true = ["rest", "rest", "fist", "fist"]
        predicted = ["rest", "fist", "fist", "fist"]
        metrics = calculate_metrics(true, predicted, ["a"] * 4, ["rest", "fist"], 0.04)
        self.assertEqual(metrics["accuracy"], accuracy_score(true, predicted))
        self.assertEqual(metrics["balanced_accuracy"], balanced_accuracy_score(true, predicted))
        self.assertEqual(metrics["macro_f1"], f1_score(true, predicted, average="macro"))
        self.assertAlmostEqual(metrics["per_class"][0]["precision"], 1)
        self.assertAlmostEqual(metrics["per_class"][0]["recall"], 0.5)

    def test_missing_rest_and_single_window_rates_are_null(self):
        metrics = calculate_metrics(["fist"], ["fist"], ["a"], ["fist"], 0.04)
        self.assertIsNone(metrics["false_activation_fraction"])
        self.assertIsNone(metrics["false_activation_episodes_per_minute"])
        self.assertIsNone(metrics["switches_per_second"])


class FixedModel:
    classes_ = np.array(["fist", "rest"])

    def predict_proba(self, features):
        return features[:, :2]


def annotated_data(probabilities, groups, truth=None, starts=None):
    if starts is None:
        starts = [i * 5 for i in range(len(groups))]
    windows = tuple(WindowInfo("s", "1", group, "gesture", i, start, start + 25,
                               start / 125, (start + 25) / 125)
                    for i, (group, start) in enumerate(zip(groups, starts)))
    labels = np.array(truth if truth is not None else ["fist", "rest"] * (len(groups) // 2))
    return BenchmarkDataset(FeatureDataset(np.array(probabilities), labels, np.array(groups)), windows)


class SplitAndSmoothingTests(unittest.TestCase):
    def test_groups_and_overlapping_sample_ranges_remain_disjoint(self):
        groups = [f"{label}_{trial}" for label in ("fist", "rest") for trial in range(4) for _ in range(3)]
        truth = [label for label in ("fist", "rest") for _ in range(12)]
        starts = [(i // 3) * 100 + (i % 3) * 5 for i in range(24)]
        dataset = annotated_data([[0.5, 0.5]] * 24, groups, truth, starts)
        train, test = grouped_split(dataset)
        self.assertFalse(set(dataset.data.groups[train]) & set(dataset.data.groups[test]))
        train_samples = {sample for i in train for sample in range(dataset.windows[i].start_sample, dataset.windows[i].end_sample)}
        test_samples = {sample for i in test for sample in range(dataset.windows[i].start_sample, dataset.windows[i].end_sample)}
        self.assertFalse(train_samples & test_samples)
        self.assertTrue(np.array_equal(train, grouped_split(dataset)[0]))

    def test_invalid_group_split_rejected(self):
        dataset = annotated_data([[0.8, 0.2]] * 4, ["a", "a", "b", "b"])
        with self.assertRaisesRegex(ValueError, "both train and test"):
            validate_split(dataset, np.array([0, 2]), np.array([1, 3]))

    def test_seven_window_mean_warmup_and_boundary_reset(self):
        probabilities = [[0.9, 0.1]] * 7 + [[0.1, 0.9], [0.1, 0.9]]
        dataset = annotated_data(probabilities, ["a"] * 8 + ["b"], ["fist"] * 7 + ["rest"] * 2)
        result = evaluate_model(FixedModel(), dataset, np.arange(9), DEFAULT_CONFIG)
        np.testing.assert_allclose(result.smoothed_probabilities[7], [5.5 / 7, 1.5 / 7])
        np.testing.assert_allclose(result.smoothed_probabilities[8], [0.1, 0.9])
        self.assertEqual(result.smoothed_predictions[7], "fist")
        self.assertEqual(result.smoothed_predictions[8], "rest")

    def test_gaps_do_not_carry_history_or_count_switches(self):
        dataset = annotated_data([[0.9, 0.1], [0.1, 0.9]], ["a", "a"], starts=[0, 10])
        result = evaluate_model(FixedModel(), dataset, np.arange(2), DEFAULT_CONFIG)
        np.testing.assert_allclose(result.smoothed_probabilities, result.raw_probabilities)
        self.assertEqual(result.metrics["raw"]["switch_count"], 0)


def create_recording(root: Path, name: str, complete=True, gestures=("fist",)) -> Path:
    directory = root / "recordings" / "S" / name
    directory.mkdir(parents=True)
    metadata = {
        "session_id": name, "subject_id": "S", "sample_rate_hz": 125,
        "channel_count": 8, "session_complete": complete, "label_timing_valid": True,
        "gestures": list(gestures), "source": "synthetic", "value_unit": "synthetic_count",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    phases = [("baseline", "rest", "")] + [phase for trial in range(1, 5)
              for gesture in gestures for phase in (("gesture", gesture, str(trial)), ("rest", "rest", str(trial)))]
    rng = np.random.default_rng(42)
    with (directory / "samples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "time_s", "frame_word", "phase", "gesture", "trial", *[f"CH{i}" for i in range(8)]])
        index = 0
        for phase, gesture, trial in phases:
            for _ in range(125):
                values = rng.normal(size=8) * (1 if gesture == "rest" else 8)
                writer.writerow([index, index / 125, "", phase, gesture, trial, *values])
                index += 1
    (directory / "events.csv").write_text(f"sample_index,event,gesture,trial\n{index},session_end,,\n", encoding="utf-8")
    return directory


def run_args(root: Path, **changes):
    values = dict(manifest=root / "data/session_manifest.csv", output=root / "results",
                  model=None, sessions=["a"], gesture_set="rest,fist", mode="within-session",
                  train_sessions=None, test_sessions=None, seed=7)
    values.update(changes)
    return argparse.Namespace(**values)


class ManifestAndIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directory = create_recording(self.root, "a")
        self.manifest = self.root / "data/session_manifest.csv"
        write_manifest(self.root / "recordings", self.manifest)

    def tearDown(self):
        self.temporary.cleanup()

    def test_manifest_and_features_match_canonical_loader(self):
        sessions, _ = select_sessions(self.manifest, ("rest", "fist"), ["a"])
        annotated = load_benchmark_dataset(sessions, DEFAULT_CONFIG)
        canonical = load_feature_dataset([self.directory], **{k: DEFAULT_CONFIG[k] for k in
                                        ("window_size", "hop_size", "thresholds", "bandpass_hz", "guard_samples", "spatial_reference", "baseline_normalize")})
        np.testing.assert_array_equal(annotated.data.features, canonical.features)
        self.assertEqual(annotated.windows[0].start_sample, 38)
        self.assertEqual(annotated.windows[0].end_sample, 63)
        self.assertEqual(len(annotated.windows), len(canonical.labels))

    def test_incomplete_and_incompatible_sessions_explicitly_excluded(self):
        create_recording(self.root, "b", complete=False)
        create_recording(self.root, "c", gestures=("pinch",))
        write_manifest(self.root / "recordings", self.manifest)
        sessions, excluded = select_sessions(self.manifest, ("rest", "fist"), ["a", "b", "c"])
        self.assertEqual([s.session_id for s in sessions], ["a"])
        self.assertIn("incomplete", excluded[0]["reason"])
        self.assertIn("incompatible", excluded[1]["reason"])

    def test_missing_file_fails_with_session_diagnostic(self):
        (self.directory / "samples.csv").unlink()
        with self.assertRaisesRegex(ValueError, "a: missing"):
            select_sessions(self.manifest, ("rest", "fist"), ["a"])

    def test_unknown_session_fails(self):
        with self.assertRaisesRegex(ValueError, "unknown session IDs"):
            select_sessions(self.manifest, ("rest", "fist"), ["missing"])

    def test_label_manifest_disagreement_fails(self):
        path = self.directory / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["gestures"] = ["pinch"]
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "recorded labels disagree"):
            select_sessions(self.manifest, ("rest", "fist"), ["a"])

    def test_complete_with_missing_flag_requires_session_end(self):
        path = self.directory / "metadata.json"
        metadata = json.loads(path.read_text())
        del metadata["session_complete"]
        path.write_text(json.dumps(metadata))
        write_manifest(self.root / "recordings", self.manifest)
        sessions, _ = select_sessions(self.manifest, ("rest", "fist"), ["a"])
        self.assertEqual(len(sessions), 1)
        with self.manifest.open(newline="") as file:
            record = next(csv.DictReader(file))
        self.assertEqual(record["recording_date"], "")
        self.assertIn("session_end", record["notes"])

    def test_end_to_end_report_is_reproducible_and_has_probabilities(self):
        result, config = run_benchmark(run_args(self.root))
        output = self.root / "results"
        for name in ("metrics.json", "run_config.json", "per_class_metrics.csv", "confusion_matrix.csv", "confusion_matrix_raw.csv", "predictions.csv", "confusion_matrix.png", "evaluation_model.joblib"):
            self.assertTrue((output / name).is_file(), name)
        first = (output / "predictions.csv").read_bytes()
        second, _ = run_benchmark(run_args(self.root))
        self.assertEqual(result.metrics, second.metrics)
        self.assertEqual(first, (output / "predictions.csv").read_bytes())
        self.assertFalse(set(config["train_groups"]) & set(config["test_groups"]))
        with (output / "predictions.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), result.metrics["test_windows"])
        self.assertIn("raw_prob_rest", rows[0])
        self.assertIn("prob_fist", rows[0])

    def test_cross_session_and_saved_model_provenance(self):
        create_recording(self.root, "b")
        write_manifest(self.root / "recordings", self.manifest)
        _, config = run_benchmark(run_args(self.root, sessions=["a", "b"], mode="cross-session",
                                          train_sessions=["a"], test_sessions=["b"]))
        self.assertEqual(config["train_session_ids"], ["a"])
        self.assertEqual(config["test_session_ids"], ["b"])
        # Store a canonical artifact fitted only on A, then test it on B.
        import joblib
        model = joblib.load(self.root / "results/evaluation_model.joblib")["model"]
        model_path = self.root / "saved.joblib"
        save_model(model_path, model, window_size=25, hop_size=5, thresholds=0.0,
                   sessions=[str(self.directory)], bandpass_hz=(10, 55), sample_rate_hz=125,
                   spatial_reference="common_average", baseline_normalize=True)
        _, saved = run_benchmark(run_args(self.root, sessions=["b"], mode="cross-session", model=model_path))
        self.assertEqual(saved["train_windows"], 0)
        self.assertEqual(saved["test_session_ids"], ["b"])
        with self.assertRaisesRegex(ValueError, "overlap its training"):
            run_benchmark(run_args(self.root, mode="cross-session", model=model_path, test_sessions=["a"]))
        with self.assertRaisesRegex(ValueError, "within-session must refit"):
            run_benchmark(run_args(self.root, model=model_path))

    def test_single_session_cross_session_fails(self):
        with self.assertRaisesRegex(ValueError, "at least two compatible"):
            run_benchmark(run_args(self.root, mode="cross-session"))

    def test_cross_session_manifest_group_must_stay_together(self):
        create_recording(self.root, "b")
        write_manifest(self.root / "recordings", self.manifest)
        with self.manifest.open(newline="") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            row["group_id"] = "one-recording"
        with self.manifest.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "at least two compatible"):
            run_benchmark(run_args(self.root, sessions=["a", "b"], mode="cross-session"))

    def test_within_condition_mismatched_posture_rejected(self):
        directory = create_recording(self.root, "b")
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["posture"] = "different"
        metadata_path.write_text(json.dumps(metadata))
        write_manifest(self.root / "recordings", self.manifest)
        with self.assertRaisesRegex(ValueError, "differ in subject, posture or source"):
            run_benchmark(run_args(self.root, sessions=["a", "b"]))

    def test_invalid_label_timing_is_excluded(self):
        metadata_path = self.directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["label_timing_valid"] = False
        metadata_path.write_text(json.dumps(metadata))
        write_manifest(self.root / "recordings", self.manifest)
        sessions, excluded = select_sessions(self.manifest, ("rest", "fist"), ["a"])
        self.assertFalse(sessions)
        self.assertIn("invalid label timing", excluded[0]["reason"])

    def test_saved_model_missing_provenance_rejected(self):
        import joblib
        run_benchmark(run_args(self.root))
        model = joblib.load(self.root / "results/evaluation_model.joblib")["model"]
        model_path = self.root / "saved.joblib"
        save_model(model_path, model, window_size=25, hop_size=5, thresholds=0.0,
                   sessions=[], bandpass_hz=(10, 55), sample_rate_hz=125,
                   spatial_reference="common_average", baseline_normalize=True)
        with self.assertRaisesRegex(ValueError, "lacks usable training-session provenance"):
            run_benchmark(run_args(self.root, mode="cross-session", model=model_path))


if __name__ == "__main__":
    unittest.main()
