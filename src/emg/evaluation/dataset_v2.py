"""Real Dataset v2 trial holdouts, posture comparisons and recalibration sweeps."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np

from ..dataset import FeatureDataset, _read_rows
from ..dataset_v2 import V2_GESTURES, V2_POSTURES, validate_v2_recording
from ..model import _new_pipeline
from .benchmark import (DEFAULT_CONFIG, _csv, _json, _sha256, _session_info,
                        evaluate_model, prediction_rows, validate_split, write_report)
from .dataset import BenchmarkDataset, load_benchmark_dataset, select_sessions
from .metrics import calculate_metrics


EXPERIMENTS = ("reference_baseline", "cross_posture", "posture_inclusive",
               "reapplication_zero_shot", "recalibration", "cross_day")


def load_v2_dataset(sessions, config=DEFAULT_CONFIG):
    """Keep canonical feature/window computation; replace splitting groups with trial IDs.

    Window group IDs retain acquisition segment boundaries for smoothing resets.
    FeatureDataset groups join all windows belonging to the same registered trial.
    """
    conditions, trials, validations = [], {}, []
    row_tables = {}
    for session in sessions:
        validations.append(validate_v2_recording(session.path))
        recorded = json.loads((session.path / "metadata.json").read_text(encoding="utf-8"))
        for key in ("electrode_placement_id", "recording_date", "electrode_reapplied", "dynamic_arm_motion"):
            if recorded[key] != session.metadata[key]:
                raise ValueError(f"{session.session_id}: manifest must not override recorded Dataset v2 {key}")
        row_tables[session.session_id] = _read_rows(session.path / "samples.csv")
        for trial in recorded["trials"]:
            if trial["trial_id"] in trials:
                raise ValueError("trial IDs must be unique across sessions")
            trials[trial["trial_id"]] = {**trial, "session_id": session.session_id}
    canonical = load_benchmark_dataset(sessions, config)
    windows, groups = [], []
    for window in canonical.windows:
        row = row_tables[window.session_id][window.start_sample]
        windows.append(replace(window, trial_id=row["trial_id"]))
        groups.append(row["trial_id"])
        conditions.append({key: row[key] for key in (
            "arm_posture", "repetition_index", "electrode_placement_id", "forearm_orientation",
            "shoulder_position", "elbow_position", "dynamic_arm_motion")})
    data = FeatureDataset(canonical.data.features, canonical.data.labels, np.asarray(groups))
    dataset = BenchmarkDataset(data, tuple(windows))
    # Any registered trial that cannot yield windows must be reported, not dropped.
    available = {w.trial_id for w in windows if w.phase == "gesture"}
    if available != set(trials):
        raise ValueError(f"registered trials have no usable guarded windows: {sorted(set(trials)-available)}")
    return dataset, conditions, trials, validations


def trial_split(dataset, indices, seed=7):
    """Half the complete trials per class; deterministic, class-balanced holdout."""
    indices = np.asarray(indices, dtype=int)
    generator = np.random.default_rng(seed)
    train_ids = set()
    for label in sorted(set(dataset.data.labels[indices])):
        groups = sorted(set(dataset.data.groups[indices][dataset.data.labels[indices] == label]))
        if len(groups) < 2:
            raise ValueError(f"{label}: need two or more complete trials for grouped holdout")
        train_ids.update(generator.permutation(groups)[:len(groups)//2].tolist())
    train = indices[np.isin(dataset.data.groups[indices], list(train_ids))]
    test = indices[~np.isin(dataset.data.groups[indices], list(train_ids))]
    validate_split(dataset, train, test)
    return train, test


def calibration_split(dataset, indices, reps, seed=7, choice=0):
    """Exactly k complete target-session trials per class; other trials test.

    Permutations are nested across k for each choice, with SeedSequence substreams.
    Zero calibration leaves all target trials held out, using the source model.
    """
    if not isinstance(reps, int) or reps < 0 or not isinstance(choice, int) or choice < 0:
        raise ValueError("calibration reps and choice must be nonnegative integers")
    indices = np.asarray(indices, dtype=int)
    generator = np.random.default_rng(np.random.SeedSequence([seed, choice]))
    selected = set()
    for label in sorted(set(dataset.data.labels[indices])):
        groups = sorted(set(dataset.data.groups[indices][dataset.data.labels[indices] == label]))
        if reps >= len(groups):
            raise ValueError(f"{label}: {reps} calibration reps would leave no held-out trial")
        selected.update(generator.permutation(groups)[:reps].tolist())
    calibration = indices[np.isin(dataset.data.groups[indices], list(selected))]
    test = indices[~np.isin(dataset.data.groups[indices], list(selected))]
    if reps:
        validate_split(dataset, calibration, test)
    return calibration, test


def independent_rest_metrics(predicted, windows, hop_seconds):
    """Window-cadence exposure and run durations, with resets at gaps/boundaries."""
    predicted = np.asarray(predicted)
    if not len(predicted):
        return {"false_active_window_fraction": None, "false_activation_episodes": 0,
                "false_activation_episodes_per_minute": None, "mean_false_activation_duration_seconds": None,
                "max_false_activation_duration_seconds": None, "switches_per_second": None,
                "rest_windows": 0, "rest_observation_seconds": 0.0}
    durations, active_run, switches, intervals = [], 0, 0, 0
    previous = None
    left_censored = right_censored = 0
    for label, window in zip(predicted, windows):
        contiguous = previous is not None and window.group_id == previous[1].group_id and abs(
            window.start_time_s - previous[1].start_time_s - hop_seconds) < 1e-5
        if not contiguous:
            if active_run:
                durations.append(active_run * hop_seconds)
                right_censored += 1
            active_run = 0
            left_censored += int(label != "rest")
        else:
            switches += int(label != previous[0])
            intervals += 1
        if label != "rest":
            active_run += 1
        elif active_run:
            durations.append(active_run * hop_seconds)
            active_run = 0
        previous = (label, window)
    if active_run:
        durations.append(active_run * hop_seconds)
        right_censored += 1
    exposure = len(predicted)*hop_seconds
    return {"false_active_window_fraction": float(np.mean(predicted != "rest")),
            "false_activation_episodes": len(durations),
            "false_activation_episodes_per_minute": len(durations)*60/exposure,
            "mean_false_activation_duration_seconds": float(np.mean(durations)) if durations else 0.0,
            "max_false_activation_duration_seconds": max(durations, default=0.0),
            "false_active_windows": int(np.sum(predicted != "rest")),
            "rest_windows": len(predicted), "rest_observation_seconds": exposure,
            "switches_per_second": switches/(intervals*hop_seconds) if intervals else None,
            "switch_count": switches, "switch_observation_seconds": intervals*hop_seconds,
            "left_censored_episodes": left_censored, "right_censored_episodes": right_censored}


def score_v2(model, dataset, classification_indices, rest_indices, config=DEFAULT_CONFIG):
    indices = np.sort(np.concatenate((classification_indices, rest_indices)))
    result = evaluate_model(model, dataset, indices, config)
    classification = np.isin(result.test_indices, classification_indices)
    rest = ~classification
    metrics = {"primary_metric": "macro_f1", "test_windows": len(indices),
               "classification_test_windows": int(np.sum(classification)),
               "independent_rest": {}, "independent_rest_by_segment": {}}
    hop = config["hop_size"]/config["sample_rate_hz"]
    for decoder, predictions in (("raw", result.raw_predictions), ("smoothed", result.smoothed_predictions)):
        subset = result.test_indices[classification]
        # Segment identifiers preserve the same smoothing and switching boundary scope.
        metrics[decoder] = calculate_metrics(dataset.data.labels[subset], predictions[classification],
                                             [dataset.windows[i].group_id for i in subset], model.classes_, hop)
        rest_windows = [dataset.windows[i] for i in result.test_indices[rest]]
        metrics["independent_rest"][decoder] = independent_rest_metrics(predictions[rest], rest_windows, hop)
        segments = {}
        for phase in dict.fromkeys(w.phase for w in rest_windows):
            mask = np.array([w.phase == phase for w in rest_windows])
            segments[phase] = independent_rest_metrics(predictions[rest][mask],
                                                       [w for w in rest_windows if w.phase == phase], hop)
        metrics["independent_rest_by_segment"][decoder] = segments
    return replace(result, metrics=metrics)


def _fit(dataset, train, test):
    validate_split(dataset, train, test)
    model = _new_pipeline(len(set(dataset.data.labels)))
    model.fit(dataset.data.features[train], dataset.data.labels[train])
    return model


def _aggregate(rows):
    grouped = {}
    for row in rows:
        key = (row["experiment"], row["test_session"], row["test_posture"], row["calibration_reps_per_gesture"], row["calibration_training"])
        grouped.setdefault(key, []).append(row)
    aggregated = []
    for members in grouped.values():
        first = members[0]
        metrics = ("macro_f1", "balanced_accuracy", "false_active_fraction", "switches_per_second")
        row = {key: first[key] for key in ("experiment", "test_session", "test_posture",
                                          "calibration_reps_per_gesture", "calibration_training")}
        row["choices"] = len(members)
        for metric in metrics:
            values = [m[metric] for m in members if m[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
            row[metric+"_std"] = float(np.std(values, ddof=1)) if len(values)>1 else (0.0 if values else None)
        aggregated.append(row)
    return aggregated


def _plots(output, summaries, aggregate, reports):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"ureca-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    posture_rows = [r for r in summaries if r["experiment"] in ("reference_baseline", "cross_posture")]
    if posture_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([r["test_posture"] for r in posture_rows], [r["macro_f1"] for r in posture_rows], "o-")
        ax.set(ylabel="Smoothed macro F1", xlabel="Static arm posture", ylim=(0, 1), title="Reference-trained cross-posture decoding")
        fig.tight_layout(); fig.savefig(plot_dir/"cross_posture.png", dpi=160); plt.close(fig)
        labels = next(iter(reports.values())).metrics["smoothed"]["labels"]
        matrix = np.array([[next(c["recall"] for c in reports[r["result_id"]].metrics["smoothed"]["per_class"] if c["label"] == label)
                            for r in posture_rows] for label in labels])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(posture_rows)), [r["test_posture"] for r in posture_rows], rotation=20)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title("Per-gesture recall by posture (reference-trained)")
        fig.colorbar(im, ax=ax, label="Recall")
        fig.tight_layout(); fig.savefig(plot_dir/"posture_recall_heatmap.png", dpi=160); plt.close(fig)
    recalibration = [r for r in aggregate if r["experiment"] == "recalibration"]
    if recalibration:
        fig, ax = plt.subplots(figsize=(7, 4))
        rows = sorted(recalibration, key=lambda r: r["calibration_reps_per_gesture"])
        ax.errorbar([r["calibration_reps_per_gesture"] for r in rows], [r["macro_f1"] for r in rows],
                    yerr=[r["macro_f1_std"] for r in rows], fmt="o-", capsize=4)
        ax.set(xlabel="Calibration repetitions per gesture", ylabel="Smoothed macro F1 (mean +/- sample SD)",
               ylim=(0, 1), title=f"Recalibration: {rows[0]['test_session']} ({rows[0]['calibration_training']})")
        ax.set_xticks([r["calibration_reps_per_gesture"] for r in rows])
        fig.tight_layout(); fig.savefig(plot_dir/"recalibration_curve.png", dpi=160); plt.close(fig)
    reference = [r for r in summaries if r["experiment"] == "reference_baseline"]
    zero = [r for r in summaries if r["experiment"] == "reapplication_zero_shot"]
    if reference and zero:
        names, values, deviations = ["Same placement", "Reapplied zero-shot"], [reference[0]["macro_f1"], zero[0]["macro_f1"]], [0, 0]
        for row in recalibration:
            if row["calibration_reps_per_gesture"]:
                names.append(f"Reapplied + {row['calibration_reps_per_gesture']} reps")
                values.append(row["macro_f1"]); deviations.append(row["macro_f1_std"])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(names, values, yerr=deviations, capsize=4)
        ax.set(ylabel="Smoothed macro F1", ylim=(0, 1), title="Placement and recalibration comparison")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout(); fig.savefig(plot_dir/"reapplication_comparison.png", dpi=160); plt.close(fig)


def run_dataset_v2_benchmark(args):
    requested = list(dict.fromkeys(i for i in (args.session_a, args.session_b, args.session_c) if i))
    labels = tuple(sorted(args.gestures))
    sessions, excluded = select_sessions(args.manifest.resolve(), labels, requested)
    if set(requested) != {s.session_id for s in sessions}:
        raise ValueError(f"requested Dataset v2 recordings excluded: {excluded}")
    if not sessions:
        raise ValueError("no complete Dataset v2 recordings selected; collect sessions first")
    identities = {s.session_id: s for s in sessions}
    a = identities[args.session_a]
    b = identities.get(args.session_b)
    c = identities.get(args.session_c)
    if a.metadata.get("session_role") != "A":
        raise ValueError("--session-a must identify a Dataset v2 Session A recording")
    for other, role in ((b, "B"), (c, "C")):
        if other:
            if other.session_id == a.session_id or other.metadata.get("session_role") != role:
                raise ValueError(f"Session {role} must have a distinct session ID and matching role")
            if other.group_id == a.group_id:
                raise ValueError("cross-session manifest groups must be independent")
            if other.metadata.get("electrode_placement_id") == a.metadata.get("electrode_placement_id"):
                raise ValueError(f"Session {role} must have a new placement ID")
            if other.metadata.get("electrode_reapplied") is not True:
                raise ValueError(f"Session {role} must declare electrode reapplication")
    if b and c and (b.session_id == c.session_id or b.metadata["electrode_placement_id"] == c.metadata["electrode_placement_id"]):
        raise ValueError("Sessions B/C must have distinct session and placement IDs")
    if b and c and b.group_id == c.group_id:
        raise ValueError("Sessions B/C manifest groups must be independent")
    if c and c.metadata.get("recording_date") == a.metadata.get("recording_date"):
        raise ValueError("cross-day Session C must have a different recording date from A")
    for key in ("subject_id", "value_unit", "sample_rate_hz", "source"):
        if len({s.metadata.get(key) for s in sessions}) != 1:
            raise ValueError(f"Dataset v2 selected sessions differ in {key}")
    validations = [validate_v2_recording(s.path) for s in sessions]
    if args.validate_only:
        return {"validation": validations}
    experiment = args.experiment.replace("-", "_")
    target = c if args.calibration_target == "C" else b
    if experiment in ("reapplication_zero_shot", "recalibration") and (b if experiment == "reapplication_zero_shot" else target) is None:
        raise ValueError("requested experiment requires its Session B/C target")
    if experiment == "cross_day" and c is None:
        raise ValueError("cross-day requires --session-c")
    if not args.calibration_reps or len(set(args.calibration_reps)) != len(args.calibration_reps) or min(args.calibration_reps) < 0 or args.choices < 1:
        raise ValueError("calibration sizes must be unique and nonnegative; choices must be positive")
    if "reference" not in args.training_postures or len(set(args.training_postures)) != len(args.training_postures):
        raise ValueError("inclusive training postures must be unique and include reference")
    output = args.output.resolve()
    project = Path(__file__).resolve().parents[3]
    for original in ("baseline_v0", "corruption_v1", "corruption_v1_1"):
        if output.is_relative_to((project / "results" / original).resolve()):
            raise ValueError("Dataset v2 output must be separate from existing benchmark reports")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory is nonempty; choose a fresh directory to avoid mixing experiments")
    dataset, conditions, trials, validations = load_v2_dataset(sessions)
    config = dict(DEFAULT_CONFIG)
    def gesture_indices(session, posture):
        return np.array([i for i,w in enumerate(dataset.windows) if w.session_id == session.session_id
                         and w.phase == "gesture" and conditions[i]["arm_posture"] == posture], dtype=int)
    def rest_indices(session):
        phases = {"rest_pre", "rest_post", *(('baseline',) if args.include_baseline_rest else ())}
        return np.array([i for i,w in enumerate(dataset.windows) if w.session_id == session.session_id and w.phase in phases], dtype=int)
    reference = gesture_indices(a, "reference")
    reference_train, reference_test = trial_split(dataset, reference, args.seed)
    reference_model = _fit(dataset, reference_train, reference_test)
    if experiment == "cross_posture" and len(a.metadata["posture_order"]) < 2:
        raise ValueError("cross-posture requires altered-posture blocks")
    if experiment == "posture_inclusive" and set(args.training_postures) - set(a.metadata["posture_order"]):
        raise ValueError("inclusive training postures missing from Session A")
    if experiment in ("all", "recalibration") and target:
        for reps in args.calibration_reps:
            calibration_split(dataset, gesture_indices(target, "reference"), reps, args.seed)
    ref_score = score_v2(reference_model, dataset, reference_test, rest_indices(a), config)
    ref_f1 = ref_score.metrics["smoothed"]["macro_f1"]
    project_sources = ("src/emg/dataset.py", "src/emg/preprocessing.py", "src/emg/features.py", "src/emg/model.py", "webapp.py",
                       "src/emg/recording.py", "src/emg/calibration.py", "src/emg/sources.py", "src/emg/session_metadata.py",
                       "src/emg/dataset_v2.py", "src/emg/evaluation/dataset_v2.py",
                       "src/emg/evaluation/dataset.py", "src/emg/evaluation/benchmark.py", "src/emg/evaluation/metrics.py", "calibrate.py")
    base_config = {"benchmark": "dataset_v2", "random_seed": args.seed, "pipeline": config,
                   "gesture_labels": list(labels), "manifest_path": str(args.manifest.resolve()),
                   "manifest_sha256": _sha256(args.manifest), "sessions_included": [_session_info(s) for s in sessions],
                   "sessions_excluded": excluded, "validation": validations,
                   "source_sha256": {name: _sha256(project/name) for name in project_sources},
                   "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in
                                      ("numpy", "scipy", "scikit-learn", "joblib", "Pillow", "matplotlib")}},
                   "latency": {"window_duration_ms": 200, "hop_ms": 40, "smoothing_windows": 7,
                               "smoothing_history_span_ms": 240, "full_smoothing_sample_span_ms": 440,
                               "measured_stabilization_delay_ms": None,
                               "note": "Configured spans, not measured reaction latency; guarded segment-local scoring omits transitions."},
                   "normalization": "unchanged canonical per-recording short baseline RMS; long rest is never used for normalization",
                   "split_unit": "stable trial_id; all windows of a trial kept together, including reconnect subsegments",
                   "classification_scope": "gesture phases only, including explicit rest trials; baseline/inter-trial rest excluded",
                   "independent_rest_scope": "rest_pre/rest_post in reference posture; baseline included only when explicitly requested",
                   "rest_duration_definition": "run length * hop_seconds; observation exposure windows * hop_seconds, excludes guards; boundary episodes can be censored",
                   "smoothing_scope": "canonical segment-local seven-window mean; histories reset at segment boundaries/gaps; ungated predictions",
                   "calibration_training": args.calibration_training,
                   "include_baseline_rest": args.include_baseline_rest,
                   "is_real_data": all(s.metadata.get("source") == "serial" for s in sessions)}
    summaries, reports = [], {}
    def save(experiment_name, name, model, train, test, session, posture, *, reps=0, choice=0, calibration=None, paired_reference=None):
        validate_split(dataset, train, test)
        rest = rest_indices(session)
        if set(dataset.data.groups[train]) & set(dataset.data.groups[rest]):
            raise ValueError("independent rest trial leaked into training")
        result = score_v2(model, dataset, test, rest, config)
        calibration_ids = set(dataset.data.groups[calibration]) if calibration is not None else set()
        active_seconds = sum(t["sample_duration_seconds"] for t in trials.values()
                             if t["trial_id"] in calibration_ids and t["gesture"] != "rest")
        all_seconds = sum(t["sample_duration_seconds"] for t in trials.values()
                          if t["trial_id"] in calibration_ids)
        smooth, independent = result.metrics["smoothed"], result.metrics["independent_rest"]["smoothed"]
        result_id = f"{experiment_name}/{name}"
        train_postures = sorted({conditions[i]["arm_posture"] for i in train})
        train_sessions = sorted({dataset.windows[i].session_id for i in train})
        train_placements = sorted({conditions[i]["electrode_placement_id"] for i in train})
        row = {"result_id": result_id, "experiment": experiment_name,
               "train_session": "|".join(train_sessions), "test_session": session.session_id,
               "train_posture": "|".join(train_postures), "test_posture": posture,
               "train_placement": "|".join(train_placements), "test_placement": session.metadata["electrode_placement_id"],
               "calibration_reps_per_gesture": reps, "calibration_choice": choice,
               "calibration_training": args.calibration_training if experiment_name == "recalibration" else "none",
               "active_calibration_seconds": active_seconds, "labelled_calibration_seconds": all_seconds,
               "gesture_count": len(labels), "accuracy": smooth["accuracy"], "macro_f1": smooth["macro_f1"],
               "delta_vs_reference": smooth["macro_f1"]-ref_f1, "balanced_accuracy": smooth["balanced_accuracy"],
               "false_active_fraction": independent["false_active_window_fraction"],
               "false_activation_episodes": independent["false_activation_episodes"],
               "false_activation_episodes_per_minute": independent["false_activation_episodes_per_minute"],
               "mean_false_activation_duration_seconds": independent["mean_false_activation_duration_seconds"],
               "max_false_activation_duration_seconds": independent["max_false_activation_duration_seconds"],
               "switches_per_second": smooth["switches_per_second"],
               "rest_switches_per_second": independent["switches_per_second"],
               "trial_rest_false_active_fraction": smooth["false_active_window_fraction"],
               "independent_rest_posture": "reference", "independent_rest_observation_seconds": independent["rest_observation_seconds"],
               "reference_only_same_test_macro_f1": paired_reference,
               "delta_vs_reference_only_same_test": smooth["macro_f1"]-paired_reference if paired_reference is not None else None,
               "is_real_data": base_config["is_real_data"]}
        run_config = {**base_config, "timestamp": datetime.now(timezone.utc).isoformat(), "experiment": experiment_name,
                      "train_window_indices": train.tolist(), "test_window_indices": test.tolist(),
                      "rest_window_indices": rest.tolist(), "train_trial_ids": sorted(set(dataset.data.groups[train])),
                      "test_trial_ids": sorted(set(dataset.data.groups[test])), "summary": row,
                      "calibration_trial_ids": sorted(set(dataset.data.groups[calibration])) if calibration is not None else [],
                      "calibration_choice_seed": [args.seed, choice],
                      "recalibration_semantics": "0: frozen A reference model; k>0: refit scaler and uniform-prior LDA on selected target trials; combined mode additionally uses A reference training partition"}
        path = output/result_id
        write_report(path, result, dataset, run_config, model)
        prediction_output = prediction_rows(result, dataset)
        test_set = set(test)
        for prediction, index in zip(prediction_output, result.test_indices):
            prediction.update(conditions[index])
            prediction["evaluation_scope"] = "classification" if index in test_set else "independent_rest"
        _csv(path/"predictions.csv", list(prediction_output[0]), prediction_output)
        _csv(path/"summary.csv", list(row), [row])
        reports[result_id] = result
        summaries.append(row)
        return result
    # Reference control is always materialized so deltas and reapplication plots are reviewable.
    save("reference_baseline", "reference", reference_model, reference_train, reference_test, a, "reference")
    if experiment in ("all", "cross_posture"):
        for posture in a.metadata["posture_order"]:
            if posture != "reference":
                test = gesture_indices(a, posture)
                save("cross_posture", posture, reference_model, reference_train, test, a, posture)
    if experiment in ("all", "posture_inclusive") and not set(args.training_postures) - set(a.metadata["posture_order"]):
        splits = {p: trial_split(dataset, gesture_indices(a, p), args.seed) for p in args.training_postures}
        train = np.sort(np.concatenate([pair[0] for pair in splits.values()]))
        first_test = splits["reference"][1]
        inclusive_model = _fit(dataset, train, first_test)
        for posture in a.metadata["posture_order"]:
            test = splits[posture][1] if posture in splits else gesture_indices(a, posture)
            paired = score_v2(reference_model, dataset, test, rest_indices(a), config).metrics["smoothed"]["macro_f1"]
            save("posture_inclusive", posture, inclusive_model, train, test, a, posture, paired_reference=paired)
    if experiment in ("all", "reapplication_zero_shot") and b:
        test = gesture_indices(b, "reference")
        save("reapplication_zero_shot", "reference", reference_model, reference_train, test, b, "reference")
    if experiment in ("all", "cross_day") and c:
        test = gesture_indices(c, "reference")
        save("cross_day", "reference", reference_model, reference_train, test, c, "reference")
    if experiment in ("all", "recalibration") and target:
        eligible = gesture_indices(target, "reference")
        for reps in sorted(args.calibration_reps):
            selections = set()
            for choice in range(1 if reps == 0 else args.choices):
                calibration, test = calibration_split(dataset, eligible, reps, args.seed, choice)
                key = tuple(sorted(set(dataset.data.groups[calibration])))
                if key in selections:
                    continue
                selections.add(key)
                if reps == 0:
                    train, model = reference_train, reference_model
                else:
                    train = np.sort(np.concatenate((reference_train, calibration))) if args.calibration_training == "combined" else calibration
                    model = _fit(dataset, train, test)
                save("recalibration", f"{target.session_id}_reps{reps}_choice{choice}", model, train, test,
                     target, "reference", reps=reps, choice=choice, calibration=calibration)
    output.mkdir(parents=True, exist_ok=True)
    aggregate = _aggregate(summaries)
    _csv(output/"summary.csv", list(summaries[0]), summaries)
    _csv(output/"summary_aggregate.csv", list(aggregate[0]), aggregate)
    skipped = []
    if experiment == "all":
        if b is None:
            skipped.append({"experiment": "reapplication_zero_shot", "reason": "Session B not supplied"})
        if c is None:
            skipped.append({"experiment": "cross_day", "reason": "Session C not supplied"})
        if target is None:
            skipped.append({"experiment": "recalibration", "reason": f"Session {args.calibration_target} not supplied"})
        if set(args.training_postures) - set(a.metadata["posture_order"]):
            skipped.append({"experiment": "posture_inclusive", "reason": "configured training posture blocks missing"})
    _json(output/"run_config.json", {**base_config, "timestamp": datetime.now(timezone.utc).isoformat(),
                                     "experiment_selection": experiment, "experiments_skipped": skipped,
                                     "calibration_target": args.calibration_target, "calibration_reps": args.calibration_reps,
                                     "choices_requested": args.choices, "training_postures": args.training_postures,
                                     "reference_train_trial_ids": sorted(set(dataset.data.groups[reference_train])),
                                     "reference_test_trial_ids": sorted(set(dataset.data.groups[reference_test])),
                                     "results": [r["result_id"] for r in summaries]})
    _plots(output, summaries, aggregate, reports)
    return {"summary": summaries, "aggregate": aggregate, "validation": validations, "experiments_skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/session_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/dataset_v2"))
    parser.add_argument("--session-a", required=True)
    parser.add_argument("--session-b")
    parser.add_argument("--session-c")
    parser.add_argument("--gestures", nargs="+", default=list(V2_GESTURES), help="exact vocabulary, including rest; no remapping")
    parser.add_argument("--experiment", choices=("all", *(name.replace("_", "-") for name in EXPERIMENTS)), default="all")
    parser.add_argument("--training-postures", nargs="+", choices=V2_POSTURES, default=list(V2_POSTURES[:3]))
    parser.add_argument("--calibration-reps", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--choices", type=int, default=5, help="deterministic calibration selections per nonzero size")
    parser.add_argument("--calibration-training", choices=("new-only", "combined"), default="new-only")
    parser.add_argument("--calibration-target", choices=("B", "C"), default="B")
    parser.add_argument("--include-baseline-rest", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        result = run_dataset_v2_benchmark(args)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(2, f"Dataset v2 error: {error}\n")
    print(json.dumps({"validation": result["validation"], "results": len(result.get("summary", [])),
                      "experiments_skipped": result.get("experiments_skipped", [])}, indent=2))
    if not args.validate_only:
        print(f"Artifacts written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
