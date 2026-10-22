"""Baseline v0: grouped held-out measurements of the existing LDA decoder."""

from __future__ import annotations

import argparse
from collections import deque
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path, PureWindowsPath
import platform
import subprocess

import joblib
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedGroupKFold

from ..features import feature_names
from ..model import _new_pipeline, load_model
from ..session_metadata import CONDITION_FIELDS
from .dataset import BenchmarkDataset, CURRENT_LABELS, Session, load_benchmark_dataset, select_sessions
from .metrics import calculate_metrics


DEFAULT_CONFIG = {
    "sample_rate_hz": 125.0, "window_size": 25, "hop_size": 5,
    "thresholds": 0.0, "bandpass_hz": (10.0, 55.0), "filter_order": 4,
    "guard_samples": 38, "spatial_reference": "common_average",
    "baseline_normalize": True, "smoothing_windows": 7,
}


@dataclass(frozen=True)
class BenchmarkResult:
    metrics: dict
    test_indices: np.ndarray
    raw_probabilities: np.ndarray
    smoothed_probabilities: np.ndarray
    raw_predictions: np.ndarray
    smoothed_predictions: np.ndarray


@dataclass(frozen=True)
class PreparedBenchmark:
    """One clean split and fitted model, shared by all subsequent stress tests."""

    result: BenchmarkResult
    run_config: dict
    model: object
    dataset: BenchmarkDataset
    sessions: tuple[Session, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray


def grouped_split(dataset: BenchmarkDataset, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Exactly the first two-fold split used by emg.model.train_and_evaluate."""
    data = dataset.data
    labels = set(data.labels)
    if len(labels) < 2 or any(len(set(data.groups[data.labels == label])) < 2 for label in labels):
        raise ValueError("need at least two independent labelled segments per class")
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    train, test = next(splitter.split(data.features, data.labels, data.groups))
    validate_split(dataset, train, test)
    return train, test


def validate_split(dataset: BenchmarkDataset, train: np.ndarray, test: np.ndarray) -> None:
    data = dataset.data
    if not len(train) or not len(test):
        raise ValueError("both train and test must contain windows")
    if set(data.groups[train]) & set(data.groups[test]):
        raise ValueError("a continuous segment appears in both train and test")
    if set(data.labels[train]) != set(data.labels) or set(data.labels[test]) != set(data.labels):
        raise ValueError("grouped split lacks classes on one side; record more trials")


def evaluate_model(model, dataset: BenchmarkDataset, test_indices: np.ndarray, config: dict) -> BenchmarkResult:
    """Score ordered held-out segments using the live deque/mean/argmax formula.

    Fresh histories at segment boundaries avoid bridging omitted prepare/guard or
    training windows. This measures the ungated classifier, not wake/hand control.
    """
    indices = np.asarray(test_indices, dtype=int)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(dataset.windows)):
        raise ValueError("test indices must be a vector of valid window indices")
    indices = np.sort(indices)
    if not len(indices) or len(set(indices.tolist())) != len(indices):
        raise ValueError("test indices must be nonempty and unique")
    if len(dataset.windows) != len(dataset.data.labels):
        raise ValueError("window provenance length differs from feature dataset")
    count = config["smoothing_windows"]
    if not isinstance(count, int) or count < 1:
        raise ValueError("smoothing_windows must be a positive integer")
    labels = tuple(str(label) for label in model.classes_)
    if set(dataset.data.labels[indices]) != set(labels):
        raise ValueError("model classes and test classes must match exactly")
    probabilities = np.asarray(model.predict_proba(dataset.data.features[indices]))
    if probabilities.shape != (len(indices), len(labels)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("model returned invalid class probabilities")
    history: deque[np.ndarray] = deque(maxlen=count)
    smoothed = []
    previous = None
    sequence_ids = []
    sequence_number = -1
    for index, probability in zip(indices, probabilities):
        window = dataset.windows[index]
        if (previous is None or window.group_id != previous.group_id
                or window.start_sample != previous.start_sample + config["hop_size"]):
            history.clear()
            sequence_number += 1
        history.append(probability)
        smoothed.append(np.mean(history, axis=0))
        sequence_ids.append(str(sequence_number))
        previous = window
    smoothed = np.vstack(smoothed)
    raw_prediction = np.asarray(labels)[np.argmax(probabilities, axis=1)]
    smooth_prediction = np.asarray(labels)[np.argmax(smoothed, axis=1)]
    truth = dataset.data.labels[indices]
    hop_seconds = config["hop_size"] / config["sample_rate_hz"]
    metrics = {
        "primary_metric": "macro_f1", "test_windows": len(indices),
        "raw": calculate_metrics(truth, raw_prediction, sequence_ids, labels, hop_seconds),
        "smoothed": calculate_metrics(truth, smooth_prediction, sequence_ids, labels, hop_seconds),
    }
    return BenchmarkResult(metrics, indices, probabilities, smoothed, raw_prediction, smooth_prediction)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, content) -> None:
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    import csv
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_confusion(metrics: dict, path: Path) -> None:
    """Small Pillow plot using the project's existing display dependency."""
    labels, matrix = metrics["labels"], metrics["confusion_matrix"]
    cell, left, top = 96, 145, 70
    canvas = Image.new("RGB", (left + cell * len(labels) + 20, top + cell * len(labels) + 45), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Smoothed decoder: rows=true, columns=predicted", fill="black")
    maximum = max(max(row) for row in matrix) or 1
    for i, label in enumerate(labels):
        draw.text((10, top + i * cell + 40), label, fill="black")
        draw.text((left + i * cell + 8, top - 25), label, fill="black")
        for j, value in enumerate(matrix[i]):
            shade = int(255 - 190 * value / maximum)
            x, y = left + j * cell, top + i * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=(shade, shade, 255), outline="white")
            draw.text((x + 35, y + 40), str(value), fill="black")
    canvas.save(path)


def prediction_rows(result: BenchmarkResult, dataset: BenchmarkDataset) -> list[dict]:
    """Shared per-window schema for clean CSVs and compressed corruption runs."""
    labels = result.metrics["raw"]["labels"]
    rows = []
    for row_index, data_index in enumerate(result.test_indices):
        window = dataset.windows[data_index]
        raw, smooth = result.raw_probabilities[row_index], result.smoothed_probabilities[row_index]
        rows.append({
            "session_id": window.session_id, "trial_id": window.trial_id,
            "group_id": window.group_id, "phase": window.phase,
            "window_index": window.window_index, "start_sample": window.start_sample,
            "end_sample": window.end_sample, "start_time_s": window.start_time_s,
            "end_time_s": window.end_time_s, "true_label": dataset.data.labels[data_index],
            "predicted_label": result.smoothed_predictions[row_index],
            "confidence": float(np.max(smooth)),
            "raw_predicted_label": result.raw_predictions[row_index],
            "raw_confidence": float(np.max(raw)),
            **{f"prob_{label}": float(value) for label, value in zip(labels, smooth)},
            **{f"raw_prob_{label}": float(value) for label, value in zip(labels, raw)},
        })
    return rows


def write_report(output: Path, result: BenchmarkResult, dataset: BenchmarkDataset, run_config: dict, model) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "metrics.json", result.metrics)
    _json(output / "run_config.json", run_config)
    per_class = [{"decoder": decoder, **row} for decoder in ("raw", "smoothed")
                 for row in result.metrics[decoder]["per_class"]]
    _csv(output / "per_class_metrics.csv", ["decoder", "label", "precision", "recall", "f1", "support"], per_class)
    labels = result.metrics["raw"]["labels"]
    for decoder, name in (("smoothed", "confusion_matrix.csv"), ("raw", "confusion_matrix_raw.csv")):
        rows = [{"true_label": label, **dict(zip(labels, row))}
                for label, row in zip(labels, result.metrics[decoder]["confusion_matrix"])]
        _csv(output / name, ["true_label", *labels], rows)
    rows = prediction_rows(result, dataset)
    _csv(output / "predictions.csv", list(rows[0]), rows)
    _plot_confusion(result.metrics["smoothed"], output / "confusion_matrix.png")
    # Separate evaluation checkpoint, never replace a deployable live model.
    joblib.dump({"model": model, "run_config": run_config}, output / "evaluation_model.joblib")


def _session_info(session: Session) -> dict:
    return {
        "session_id": session.session_id, "path": str(session.path), "group_id": session.group_id,
        "metadata": session.metadata,
        "sha256": {name: _sha256(session.path / name) for name in
                   ("metadata.json", "samples.csv", "events.csv")},
    }


def _saved_training_sessions(artifact: dict, sessions: list[Session], root: Path) -> tuple[set[str], list[str]]:
    references = artifact.get("sessions")
    if not isinstance(references, list) or not references or not all(isinstance(item, str) and item for item in references):
        raise ValueError("saved model lacks usable training-session provenance; cannot establish held-out data")
    training_ids = set()
    for reference in references:
        # Older artifacts use Windows relative paths; matching IDs also survives relocation.
        basename = PureWindowsPath(reference.replace("/", "\\")).name
        resolved = (root / Path(reference.replace("\\", "/"))).resolve()
        for session in sessions:
            if session.path == resolved or session.path.name == basename or session.session_id == reference:
                training_ids.add(session.session_id)
    return training_ids, references


def prepare_benchmark(args: argparse.Namespace) -> PreparedBenchmark:
    manifest = args.manifest.resolve()
    config = dict(DEFAULT_CONFIG)
    artifact = load_model(args.model) if args.model else None
    if artifact:
        if args.mode != "cross-session" or args.train_sessions:
            raise ValueError("--model requires --mode cross-session, without --train-sessions; within-session must refit held-out groups")
        for key in ("window_size", "hop_size", "thresholds", "sample_rate_hz"):
            config[key] = artifact[key]
        config.update({
            "bandpass_hz": artifact.get("bandpass_hz"),
            "spatial_reference": artifact.get("spatial_reference", "none"),
            "baseline_normalize": bool(artifact.get("baseline_normalize", False)),
        })
        if artifact.get("feature_names") != feature_names():
            raise ValueError("saved model feature order differs from canonical features")
    labels = tuple(sorted(label.strip() for label in args.gesture_set.split(","))) if args.gesture_set else (
        tuple(str(label) for label in artifact["model"].classes_) if artifact else tuple(sorted(CURRENT_LABELS))
    )
    if len(labels) < 2 or len(set(labels)) != len(labels) or any(not label for label in labels):
        raise ValueError("gesture-set must contain at least two distinct nonempty labels")
    if artifact and set(labels) != set(artifact["model"].classes_):
        raise ValueError("requested gesture set differs from saved model classes")
    if args.mode == "within-session" and (args.train_sessions or args.test_sessions):
        raise ValueError("explicit --train-sessions/--test-sessions require cross-session mode")
    requested = args.sessions
    if not requested and (args.train_sessions or args.test_sessions):
        requested = list(dict.fromkeys((args.train_sessions or []) + (args.test_sessions or [])))
    metadata_filters = {}
    for expression in getattr(args, "where", None) or []:
        key, separator, value = expression.partition("=")
        if not separator or not key.strip() or not value.strip() or key.strip() in metadata_filters:
            raise ValueError("--where requires distinct FIELD=VALUE expressions")
        metadata_filters[key.strip()] = value.strip()
    sessions, excluded = select_sessions(manifest, labels, requested, metadata_filters)
    for exclusion in excluded:
        print(f"Excluded {exclusion['session_id']}: {exclusion['reason']}")
    if not sessions:
        raise ValueError("no compatible complete sessions selected")
    if requested and set(requested) - {session.session_id for session in sessions}:
        raise ValueError("one or more explicitly requested sessions were excluded; see diagnostics")
    for session in sessions:
        if float(session.metadata["sample_rate_hz"]) != float(config["sample_rate_hz"]):
            raise ValueError(f"{session.session_id}: sample rate differs from model/config")
    if len({session.metadata.get("value_unit") for session in sessions}) != 1:
        raise ValueError("selected sessions have incompatible signal units")
    if args.mode == "within-session" and any(
        len({session.metadata.get(key) for session in sessions}) != 1
        for key in ("subject_id", "posture", "source", *CONDITION_FIELDS)
    ):
        raise ValueError("within-condition sessions differ in subject, posture or source; select comparable sessions or use cross-session mode")
    training_references = []
    if artifact:
        training_ids, training_references = _saved_training_sessions(artifact, sessions, manifest.parent.parent)
        if args.test_sessions and set(args.test_sessions) & training_ids:
            raise ValueError("saved model test sessions overlap its training provenance")
        testing_ids = set(args.test_sessions or [session.session_id for session in sessions
                                                if session.session_id not in training_ids])
        for session in sessions:
            if session.session_id in training_ids:
                excluded.append({"session_id": session.session_id, "reason": "saved model training session; excluded from held-out testing"})
                print(f"Excluded {session.session_id}: saved model training session")
        sessions = [session for session in sessions if session.session_id in testing_ids]
        if not sessions:
            raise ValueError("no unseen compatible test sessions; saved model was fitted on the available sessions")
    dataset = load_benchmark_dataset(sessions, config)
    if artifact:
        train = np.array([], dtype=int)
        test = np.arange(len(dataset.windows))
        model = artifact["model"]
        split_method = "saved model; test sessions disjoint from recorded training-session provenance"
    elif args.mode == "within-session":
        train, test = grouped_split(dataset, args.seed)
        # Import the canonical pipeline including its uniform class priors.
        model = _new_pipeline(len(labels))
        model.fit(dataset.data.features[train], dataset.data.labels[train])
        split_method = "StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed); first fold; continuous labelled segments"
    else:
        if len({session.group_id for session in sessions}) < 2:
            raise ValueError("cross-session baseline unavailable: need at least two compatible independent session groups")
        all_ids = {session.session_id for session in sessions}
        testing_ids = set(args.test_sessions or [sessions[-1].session_id])
        training_ids = set(args.train_sessions or (all_ids - testing_ids))
        if training_ids & testing_ids or training_ids | testing_ids != all_ids:
            raise ValueError("cross-session train/test IDs must be disjoint and cover all selected sessions")
        training_groups = {session.group_id for session in sessions if session.session_id in training_ids}
        testing_groups = {session.group_id for session in sessions if session.session_id in testing_ids}
        if training_groups & testing_groups:
            raise ValueError("manifest group_id appears in both train and test sessions")
        train = np.array([i for i, window in enumerate(dataset.windows) if window.session_id in training_ids])
        test = np.array([i for i, window in enumerate(dataset.windows) if window.session_id in testing_ids])
        validate_split(dataset, train, test)
        model = _new_pipeline(len(labels))
        model.fit(dataset.data.features[train], dataset.data.labels[train])
        split_method = "whole-session holdout; explicit session IDs, or last manifest session held out by default"
    result = evaluate_model(model, dataset, test, config)
    project = Path(__file__).resolve().parents[3]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True, check=False)
    source_names = ("src/emg/dataset.py", "src/emg/preprocessing.py", "src/emg/features.py", "src/emg/model.py", "webapp.py",
                    "src/emg/evaluation/benchmark.py", "src/emg/evaluation/dataset.py", "src/emg/evaluation/metrics.py")
    run_config = {
        "benchmark": "baseline_v0", "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": args.seed, "mode": args.mode, "pipeline": config,
        "feature_list": feature_names(), "gesture_labels": list(labels), "channel_count": 8,
        "model_path": str(args.model.resolve()) if args.model else None,
        "model_sha256": _sha256(args.model) if args.model else None,
        "model_training_session_references": training_references,
        "classifier": "canonical StandardScaler + LDA (uniform priors when refitting)",
        "manifest_path": str(manifest), "manifest_sha256": _sha256(manifest),
        "metadata_filters": metadata_filters,
        "sessions_included": [_session_info(session) for session in sessions], "sessions_excluded": excluded,
        "train_session_ids": sorted({dataset.windows[i].session_id for i in train}),
        "test_session_ids": sorted({dataset.windows[i].session_id for i in test}),
        "train_groups": sorted(set(dataset.data.groups[train])), "test_groups": sorted(set(dataset.data.groups[test])),
        "train_window_indices": train.tolist(), "test_window_indices": test.tolist(),
        "train_windows": len(train), "test_windows": len(test), "split_method": split_method,
        "train_phase_window_counts": dict(Counter(dataset.windows[i].phase for i in train)),
        "test_phase_window_counts": dict(Counter(dataset.windows[i].phase for i in test)),
        "smoothing": "live deque(maxlen=7), mean probabilities, argmax; partial deque during warmup; reset at each held-out segment or gap",
        "measurement_scope": "ungated raw LDA and segment-local smoothed predictions; no wake sequence, hold detector, or 3D interpolation",
        "timing": "hop_size/sample_rate_hz; CSV time_s is sample-index-derived, not hardware wall time",
        "switch_rate_definition": "within-sequence switches / sum((windows_in_sequence-1)*hop_seconds); no boundary switches",
        "false_activation_definition": "episodes start on first false-active rest window, including left-censored starts; non-rest class changes within one episode do not add episodes",
        "rest_rate_definition": "episodes*60/(rest_windows*hop_seconds); approximate window-cadence exposure, excluding guards and omitted intervals",
        "normalization_scope": "per-session initial rest RMS, including evaluation sessions; baseline rest windows remain eligible as in canonical loader",
        "cross_session_available_for_selected_sessions": len({s.group_id for s in sessions}) >= 2,
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "source_sha256": {name: _sha256(project / name) for name in source_names},
        "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "scikit-learn", "joblib", "Pillow")}},
    }
    return PreparedBenchmark(result, run_config, model, dataset, tuple(sessions), train, test)


def run_benchmark(args: argparse.Namespace) -> tuple[BenchmarkResult, dict]:
    prepared = prepare_benchmark(args)
    write_report(args.output, prepared.result, prepared.dataset, prepared.run_config, prepared.model)
    return prepared.result, prepared.run_config


def print_summary(result: BenchmarkResult, run_config: dict, output: Path) -> None:
    print("\nBaseline Benchmark v0")
    print("Sessions included: " + ", ".join(s["session_id"] for s in run_config["sessions_included"]))
    print("Gesture classes: " + ", ".join(run_config["gesture_labels"]))
    print("Evaluation: " + run_config["split_method"])
    print(f"Windows: {run_config['train_windows']} training, {run_config['test_windows']} held out")
    for decoder in ("raw", "smoothed"):
        metrics = result.metrics[decoder]
        print(f"\n{decoder.capitalize()} decoder:")
        for name in ("accuracy", "balanced_accuracy", "macro_f1", "switches_per_second",
                     "false_active_windows", "rest_windows", "false_active_window_fraction",
                     "false_activation_episodes", "false_activation_episodes_per_minute"):
            value = metrics[name]
            print(f"  {name}: {value:.6f}" if isinstance(value, float) else f"  {name}: {value}")
    print("\nSmoothed confusion matrix (rows=true, columns=predicted):")
    print("Labels: " + ", ".join(result.metrics["smoothed"]["labels"]))
    print(np.array(result.metrics["smoothed"]["confusion_matrix"]))
    if run_config["mode"] == "within-session" and not run_config["cross_session_available_for_selected_sessions"]:
        print("Cross-session baseline unavailable for this selection: fewer than two compatible session groups.")
    print(f"Artifacts written to: {output.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/session_manifest.csv"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--sessions", nargs="+")
    parser.add_argument("--gesture-set", help="comma-separated exact classes, including rest; default is current browser set")
    parser.add_argument("--mode", choices=("within-session", "cross-session", "corruption"), default="within-session")
    parser.add_argument("--corruption", choices=("all", "none", "broadband", "powerline", "channel-dropout", "channel-noise"), default="all")
    parser.add_argument("--repeats", type=int, default=5, help="stochastic realizations per condition in corruption mode")
    parser.add_argument("--include-pair-dropout", action="store_true")
    parser.add_argument("--corruption-version", choices=("v1", "v1.1"), default="v1.1",
                        help="v1.1 uses AC scaling; v1 explicitly reproduces raw-power scaling")
    parser.add_argument("--channel-noise-snr", type=float, nargs="+", help="v1.1 single-channel SNR sweep; e.g. 20 15 10 5 0")
    parser.add_argument("--where", action="append", metavar="FIELD=VALUE", help="exact manifest/condition filter; repeat for AND")
    parser.add_argument("--train-sessions", nargs="+")
    parser.add_argument("--test-sessions", nargs="+")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(("results/corruption_v1_1" if args.corruption_version == "v1.1" else
                            "results/corruption_v1_reproduction") if args.mode == "corruption" else "results/baseline_v0")
    try:
        if args.mode == "corruption":
            from .corruption_benchmark import run_corruption_benchmark
            run_corruption_benchmark(args)
            return 0
        result, config = run_benchmark(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Benchmark error: {error}\n")
    print_summary(result, config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
