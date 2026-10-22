"""Corruption sweeps sharing Baseline v0's clean split, fitted model and metrics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import io
import os
from pathlib import Path
import pickle
import tempfile

import joblib
import numpy as np

from .benchmark import (
    _csv, _json, _plot_confusion, _sha256, evaluate_model, prediction_rows,
    prepare_benchmark,
)
from .corruptions import ac_signal_power, resolved_corruption_config, signal_power, sweep_conditions
from .dataset import corrupt_heldout_dataset, load_raw_recordings


METRIC_FIELDS = {
    "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "macro_f1",
    "switch_count": "switch_count", "switches_per_second": "switches_per_second",
    "false_active_fraction": "false_active_window_fraction",
    "false_active_windows": "false_active_windows", "rest_windows": "rest_windows",
    "false_activation_episode_count": "false_activation_episodes",
    "false_activation_episodes_per_minute": "false_activation_episodes_per_minute",
}
DELTA_FIELDS = {
    "macro_f1": "macro_f1", "balanced_accuracy": "balanced_accuracy",
    "false_active_fraction": "false_active_window_fraction", "switch_rate": "switches_per_second",
}
SPLIT_SEED = 7


def _mean_std(values):
    if all(value is None for value in values):
        return None, None
    if any(value is None for value in values):
        raise ValueError("metric availability differs across repeats")
    values = np.array(values, dtype=float)
    if np.all(values == values[0]):
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def aggregate_condition(condition, runs: list[dict], clean: dict, seeds: list[int]) -> dict:
    row = {
        "condition": condition.condition_id, "corruption": condition.corruption.name,
        "severity": condition.severity, "channel": condition.channel,
        "seed": seeds[0] if condition.corruption.stochastic else None,
        "seeds": "|".join(map(str, seeds)) if condition.corruption.stochastic else None,
        "repeats": len(runs),
    }
    for decoder in ("raw", "smoothed"):
        for column, metric in METRIC_FIELDS.items():
            mean, std = _mean_std([run[decoder][metric] for run in runs])
            row[f"{decoder}_{column}"] = mean
            row[f"{decoder}_{column}_std"] = std
        for column, metric in DELTA_FIELDS.items():
            baseline = clean[decoder][metric]
            changes = [None if baseline is None else run[decoder][metric] - baseline for run in runs]
            mean, std = _mean_std(changes)
            row[f"delta_{decoder}_{column}"] = mean
            row[f"delta_{decoder}_{column}_std"] = std
    return row


def _save_predictions(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    # Fixed gzip header timestamp and no embedded output filename: byte reproducibility.
    path.write_bytes(gzip.compress(buffer.getvalue().encode("utf-8"), mtime=0))


def _save_run(output: Path, run_id: str, condition, repeat: int, seed, result, dataset, audit) -> None:
    directory = output / condition.corruption.name / run_id
    directory.mkdir(parents=True, exist_ok=True)
    _json(directory / "metrics.json", {
        "condition": condition.condition_id, "corruption": asdict(condition.corruption),
        "repeat": repeat, "seed": seed, **result.metrics,
    })
    _json(directory / "signal_audit.json", audit)
    for decoder in ("raw", "smoothed"):
        metrics = result.metrics[decoder]
        labels = metrics["labels"]
        matrix_rows = [{"true_label": label, **dict(zip(labels, row))}
                       for label, row in zip(labels, metrics["confusion_matrix"])]
        _csv(directory / f"confusion_matrix_{decoder}.csv", ["true_label", *labels], matrix_rows)
    class_rows = [{"decoder": decoder, **row} for decoder in ("raw", "smoothed")
                  for row in result.metrics[decoder]["per_class"]]
    _csv(directory / "per_class_metrics.csv", ["decoder", "label", "precision", "recall", "f1", "support"], class_rows)
    _save_predictions(output / "predictions" / f"{run_id}.csv.gz", prediction_rows(result, dataset))


def _class_summary(condition_id: str, runs: list[dict], clean: dict) -> list[dict]:
    records = []
    for decoder in ("raw", "smoothed"):
        for index, label in enumerate(clean[decoder]["labels"]):
            baseline = clean[decoder]["per_class"][index]
            row = {"condition": condition_id, "decoder": decoder, "label": label,
                   "support": baseline["support"], "repeats": len(runs)}
            for metric in ("precision", "recall", "f1"):
                mean, std = _mean_std([run[decoder]["per_class"][index][metric] for run in runs])
                row[metric], row[f"{metric}_std"] = mean, std
                row[f"delta_{metric}"] = mean - baseline[metric]
            records.append(row)
    return records


def _plots(output: Path, summary: list[dict], power_reference: str) -> None:
    # Keep font caches in writable temporary storage, independent of the user's home.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ureca-matplotlib"))
    try:
        import matplotlib
    except ImportError as error:
        raise ValueError("corruption plots require requirements-benchmark.txt; install it with pip -r") from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = summary[0]
    reference_label = "AC" if power_reference == "ac" else "raw"
    colors = {"raw": "#b45f06", "smoothed": "#225ea8"}
    for family, name, title, xlabel in (
        ("broadband_noise", "broadband_snr_curve.png", f"Broadband noise ({reference_label} scaling)", f"Target {reference_label} mean-square SNR (dB)"),
        ("powerline_50hz", "powerline_50hz_curve.png", f"50 Hz interference ({reference_label} scaling)", f"Sine peak amplitude / {reference_label} channel RMS"),
    ):
        rows = [clean] + [row for row in summary if row["corruption"] == family]
        if len(rows) == 1:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
        x = np.arange(len(rows))
        for decoder in ("raw", "smoothed"):
            ax.errorbar(x, [row[f"{decoder}_macro_f1"] for row in rows],
                        yerr=[row[f"{decoder}_macro_f1_std"] for row in rows],
                        marker="o", capsize=4, label=decoder, color=colors[decoder])
        ax.set_xticks(x, ["clean"] + [f"{row['severity']:g}" for row in rows[1:]])
        ax.set(xlabel=xlabel, ylabel="Macro F1 (mean ± sample SD)", title=title, ylim=(0, 1))
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.savefig(output / "plots" / name, dpi=150)
        plt.close(fig)
    for family, name, title in (
        ("channel_dropout", "channel_dropout.png", "Single-channel dropout before CAR"),
        ("single_channel_noise", "single_channel_noise.png", f"Single-channel noise ({reference_label} scaling), lowest SNR"),
    ):
        rows = [row for row in summary if row["corruption"] == family and "|" not in row["channel"]]
        if family == "single_channel_noise" and rows:
            lowest_snr = min(row["severity"] for row in rows)
            rows = [row for row in rows if row["severity"] == lowest_snr]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.bar([row["channel"] for row in rows], [row["smoothed_macro_f1"] for row in rows],
               yerr=[row["smoothed_macro_f1_std"] for row in rows], capsize=4, color=colors["smoothed"])
        ax.axhline(clean["smoothed_macro_f1"], color="#333333", linestyle="--", label="clean")
        ax.set(xlabel="Physical channel (CH1 = recording CH0)", ylabel="Smoothed macro F1 (mean ± sample SD)",
               title=title, ylim=(0, 1))
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.savefig(output / "plots" / name, dpi=150)
        plt.close(fig)
    noise_rows = [row for row in summary if row["corruption"] == "single_channel_noise"]
    levels = list(dict.fromkeys(row["severity"] for row in noise_rows))
    if len(levels) > 1:
        channels = [f"CH{ch + 1}" for ch in range(8)]
        values = np.array([[next(row["smoothed_macro_f1"] for row in noise_rows
                                if row["channel"] == channel and row["severity"] == level)
                            for level in levels] for channel in channels])
        fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
        color = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(levels)), [f"{level:g}" for level in levels])
        ax.set_yticks(range(8), channels)
        for i in range(8):
            for j in range(len(levels)):
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center",
                        color="black" if values[i, j] > 0.6 else "white")
        ax.set(xlabel=f"Single-channel {reference_label} SNR (dB)", ylabel="Channel (CH1 = CSV CH0)",
               title="Smoothed macro F1 (mean across repeats)")
        fig.colorbar(color, ax=ax, label="Macro F1")
        fig.savefig(output / "plots/single_channel_noise_heatmap.png", dpi=150)
        plt.close(fig)


def signal_scale_diagnostics(recordings, heldout_groups: set[str]) -> tuple[list[dict], dict]:
    rows = []
    for recording in recordings:
        for segment, (start, end) in enumerate(recording.segments):
            values = recording.samples[start:end]
            mean = np.mean(values, axis=0)
            raw_rms = np.sqrt(signal_power(values))
            ac_rms = np.sqrt(ac_signal_power(values))
            group = f"{recording.session.path.resolve()}::segment_{segment}"
            for ch in range(8):
                rows.append({"session_id": recording.session.session_id,
                             "segment_id": segment, "channel": f"CH{ch + 1}",
                             "phase": recording.rows[start]["phase"], "gesture": recording.rows[start]["gesture"],
                             "start_sample": int(recording.rows[start]["sample_index"]),
                             "end_sample": int(recording.rows[end - 1]["sample_index"]) + 1,
                             "held_out": group in heldout_groups, "raw_mean": float(mean[ch]),
                             "raw_rms": float(raw_rms[ch]), "ac_rms": float(ac_rms[ch]),
                             "raw_to_ac_rms_ratio": float(raw_rms[ch] / ac_rms[ch]) if ac_rms[ch] > 0 else None})
    statistics = {}
    for name, selected in (("all_segments", rows), ("heldout_segments", [row for row in rows if row["held_out"]])):
        ratios = [row["raw_to_ac_rms_ratio"] for row in selected if row["raw_to_ac_rms_ratio"] is not None]
        statistics[name] = {"segment_channel_rows": len(selected), "zero_ac_rows": sum(row["ac_rms"] == 0 for row in selected),
                            **{f"median_{key}": float(np.median([row[key] for row in selected])) if selected else None
                               for key in ("raw_mean", "raw_rms", "ac_rms")},
                            "median_ratio": float(np.median(ratios)) if ratios else None,
                            "min_ratio": float(min(ratios)) if ratios else None,
                            "max_ratio": float(max(ratios)) if ratios else None}
    return rows, statistics


def run_corruption_benchmark(args: argparse.Namespace) -> tuple[list[dict], dict]:
    if args.repeats < 1 or args.seed < 0:
        raise ValueError("repeats must be positive and corruption seed non-negative")
    if args.model or args.train_sessions or args.test_sessions:
        raise ValueError("corruption v1 freezes the within-session split; use clean recordings, without --model or explicit train/test sessions")
    version = getattr(args, "corruption_version", "v1.1")
    channel_snrs = getattr(args, "channel_noise_snr", None)
    sweep_config = resolved_corruption_config(version, channel_snrs)
    conditions = sweep_conditions(args.corruption, args.include_pair_dropout, version, channel_snrs)
    # Noise seed never changes the reference split. Fit exactly once, on clean features.
    clean_args = argparse.Namespace(**{**vars(args), "mode": "within-session", "seed": SPLIT_SEED})
    prepared = prepare_benchmark(clean_args)
    clean = prepared.result.metrics
    recordings = load_raw_recordings(prepared.sessions)
    model_digest = hashlib.sha256(pickle.dumps(prepared.model, protocol=5)).hexdigest()
    training_digest = hashlib.sha256(prepared.dataset.data.features[prepared.train_indices].tobytes()).hexdigest()
    output = args.output.resolve()
    project_results = Path(__file__).resolve().parents[3] / "results"
    protected = (project_results / "baseline_v0", project_results / "corruption_v1")
    if any(output.is_relative_to(path.resolve()) or output == Path("results", path.name).resolve() for path in protected):
        raise ValueError("corruption output must be separate from original baseline_v0 and corruption_v1 reports")
    for name in ("predictions", "plots"):
        (output / name).mkdir(parents=True, exist_ok=True)
    scale_rows, scale_statistics = signal_scale_diagnostics(recordings, set(prepared.dataset.data.groups[prepared.test_indices]))
    _csv(output / "signal_scale_summary.csv", list(scale_rows[0]), scale_rows)
    _json(output / "signal_scale_statistics.json", scale_statistics)
    summary, per_class, run_registry = [], [], []
    # Retain only selected confusion candidates; do not duplicate checkpoint files.
    visual_candidates = {}
    for condition in conditions:
        seeds = [args.seed + repeat for repeat in range(args.repeats)] if condition.corruption.stochastic else []
        run_metrics = []
        for repeat in range(len(seeds) if seeds else 1):
            seed = seeds[repeat] if seeds else None
            run_id = condition.condition_id + (f"_seed{seed}" if seed is not None else "")
            if condition.corruption.name == "clean":
                dataset, audit, result = prepared.dataset, [], prepared.result
            else:
                dataset, audit = corrupt_heldout_dataset(
                    prepared.dataset, recordings, prepared.test_indices, prepared.run_config["pipeline"],
                    condition.corruption, seed if seed is not None else args.seed,
                )
                if not np.array_equal(dataset.data.features[prepared.train_indices], prepared.dataset.data.features[prepared.train_indices]):
                    raise ValueError("corruption changed training features")
                result = evaluate_model(prepared.model, dataset, prepared.test_indices, prepared.run_config["pipeline"])
            _save_run(output, run_id, condition, repeat, seed, result, dataset, audit)
            run_metrics.append(result.metrics)
            run_registry.append({"run_id": run_id, "condition": condition.condition_id, "repeat": repeat,
                                 "seed": seed, "metrics_path": f"{condition.corruption.name}/{run_id}/metrics.json",
                                 "predictions_path": f"predictions/{run_id}.csv.gz"})
            if repeat == 0:
                visual_candidates[condition.condition_id] = result.metrics["smoothed"]
        summary.append(aggregate_condition(condition, run_metrics, clean, seeds))
        per_class.extend(_class_summary(condition.condition_id, run_metrics, clean))
        print(f"{condition.condition_id}: smoothed macro F1 {summary[-1]['smoothed_macro_f1']:.6f} "
              f"+/- {summary[-1]['smoothed_macro_f1_std']:.6f} ({len(run_metrics)} run(s))", flush=True)
    if model_digest != hashlib.sha256(pickle.dumps(prepared.model, protocol=5)).hexdigest():
        raise ValueError("frozen model changed during corruption evaluation")
    if training_digest != hashlib.sha256(prepared.dataset.data.features[prepared.train_indices].tobytes()).hexdigest():
        raise ValueError("clean training dataset changed during corruption evaluation")
    worst = {}
    for family in ("channel_dropout", "single_channel_noise"):
        rows = [row for row in summary if row["corruption"] == family and "|" not in row["channel"]]
        if rows:
            worst[family] = min(rows, key=lambda row: row["smoothed_macro_f1"])["condition"]
    powerline_rows = [row for row in summary if row["corruption"] == "powerline_50hz"]
    highest_powerline = max(powerline_rows, key=lambda row: row["severity"])["condition"] if powerline_rows else ""
    selected_visuals = ["clean", "broadband_snr0", highest_powerline, *worst.values()]
    for condition_id in selected_visuals:
        if condition_id in visual_candidates:
            _plot_confusion(visual_candidates[condition_id], output / "plots" / f"confusion_{condition_id}.png")
    _plots(output, summary, sweep_config["power_reference"])
    _csv(output / "summary.csv", list(summary[0]), summary)
    _csv(output / "per_class_summary.csv", list(per_class[0]), per_class)
    for decoder in ("raw", "smoothed"):
        labels = clean[decoder]["labels"]
        rows = []
        for condition in conditions:
            values = {row["label"]: row for row in per_class if row["condition"] == condition.condition_id and row["decoder"] == decoder}
            rows.append({"condition": condition.condition_id,
                         **{label: values[label]["recall"] for label in labels},
                         **{f"delta_{label}": values[label]["delta_recall"] for label in labels},
                         **{f"std_{label}": values[label]["recall_std"] for label in labels}})
        _csv(output / f"recall_changes_{decoder}.csv", list(rows[0]), rows)
    damaged = [row for row in summary if row["condition"] != "clean"]
    worst_condition = min(damaged, key=lambda row: row["smoothed_macro_f1"]) if damaged else None
    affected_rows = [row for row in per_class if row["decoder"] == "smoothed" and row["condition"] != "clean"]
    affected = min(affected_rows, key=lambda row: row["delta_recall"]) if affected_rows else None
    class_mean_delta = [{"label": label, "mean_delta_recall": float(np.mean([
        row["delta_recall"] for row in affected_rows if row["label"] == label
    ]))} for label in clean["smoothed"]["labels"]] if affected_rows else []
    worst_ties = [row["condition"] for row in damaged if worst_condition and np.isclose(
        row["smoothed_macro_f1"], worst_condition["smoothed_macro_f1"], atol=1e-12, rtol=0
    )]
    findings = {"worst_condition": worst_condition, "worst_channels": worst,
                "worst_conditions_tied": worst_ties,
                "largest_smoothed_recall_drop": affected,
                "class_mean_recall_deltas": class_mean_delta,
                "most_affected_gesture_by_mean_recall_delta": min(class_mean_delta, key=lambda row: row["mean_delta_recall"]) if class_mean_delta else None,
                "class_ranking_definition": "mean recall change with equal weight per non-clean condition after averaging repeats"}
    noise_summary = [row for row in summary if row["corruption"] == "single_channel_noise"]
    findings["worst_noisy_channel_by_snr"] = [
        {"snr_db": snr, "channel": min([row for row in noise_summary if row["severity"] == snr],
                                       key=lambda row: row["smoothed_macro_f1"])["channel"]}
        for snr in dict.fromkeys(row["severity"] for row in noise_summary)
    ]
    _json(output / "summary.json", {"primary_metric": "smoothed_macro_f1", "conditions": summary,
                                    "runs": run_registry, "findings": findings})
    source_root = Path(__file__).resolve().parents[3]
    run_config = {
        **prepared.run_config, "benchmark": "corruption_v1_1" if version == "v1.1" else "corruption_v1", "mode": "corruption",
        "corruption_version": version,
        "clean_split_seed": SPLIT_SEED, "corruption_seed": args.seed, "repeats": args.repeats,
        "corruption_config": sweep_config, "corruption_selection": args.corruption,
        "signal_scale_statistics": scale_statistics,
        "include_pair_dropout": args.include_pair_dropout,
        "conditions": [{"condition": condition.condition_id, **asdict(condition.corruption)} for condition in conditions],
        "runs": run_registry, "fitted_model_pickle_sha256": model_digest, "clean_train_features_sha256": training_digest,
        "seed_derivation": "repeat seed=base_seed+repeat; segment Generator via SeedSequence([repeat_seed, four big-endian uint32 words of SHA256(session_id)[:16], segment_number]); matched substreams across severity levels",
        "injection_location": "raw recorded CH0..CH7 unscaled ADC counts, before CAR/filter/normalization/features; full held-out labelled segments including guards",
        "corruption_scope": "only raw segments represented in the frozen test groups; training and prepare raw samples remain clean",
        "normalization_scope": "unchanged loader: RMS of all filtered initial baseline samples per session; baseline itself corrupted iff its segment is held out; scaler/LDA remain fitted to clean training features",
        "continuous_filter_scope": "same full-recording causal filter as v0; no resets added; output features at train indices discarded and replaced by original clean features",
        "power_definition": ("per-segment per-channel mean((raw_count-mean(raw_count))**2); AC scaling ONLY, clean raw input is not centered before the decoder"
                             if version == "v1.1" else "per-segment per-channel mean(raw_count**2), INCLUDING DC; legacy v1 convention"),
        "zero_power_behavior": "zero/underflow channel power => zero added Gaussian/sine noise; no noise floor, no division by signal power",
        "powerline_amplitude_definition": f"A=multiplier*{sweep_config['power_reference']}_channel_RMS is sine peak amplitude, not injected RMS; independent phase per channel and segment",
        "channel_numbering": "reports CH1..CH8 correspond to CSV CH0..CH7 and API zero-based indices 0..7",
        "aggregate_definition": "means and sample standard deviations (ddof=1) over stochastic repeats; deterministic/single-run SD=0; no prediction averaging",
        "confusion_visuals": "conditions chosen by aggregate smoothed macro F1; plotted matrices are first realization (lowest seed), not summed or mean matrices",
        "limitations": ["Synthetic stressors do not perfectly reproduce electrode displacement, motion artefact or skin-electrode impedance changes.",
                       ("AC scale is estimated from the entire mean-removed segment, not specifically post-filter 10-55 Hz EMG; drift or artefacts may contribute."
                        if version == "v1.1" else "Raw DC offset contributes to specified SNR/RMS; corruption can greatly exceed useful AC EMG even at 20 dB."),
                       "Held-out baseline also supplies normalization, retaining v0's calibration/evaluation coupling.",
                       "Current gesture set has one recorded session; this is grouped segment holdout, not independent-session validation."],
        "environment": {**prepared.run_config["environment"], "matplotlib": importlib.metadata.version("matplotlib")},
        "source_sha256": {**prepared.run_config["source_sha256"], **{
            name: _sha256(source_root / name) for name in
            ("src/emg/evaluation/corruptions.py", "src/emg/evaluation/corruption_benchmark.py", "src/emg/session_metadata.py")}},
    }
    _json(output / "run_config.json", run_config)
    joblib.dump({"model": prepared.model, "run_config": run_config}, output / "evaluation_model.joblib")
    print(f"Artifacts written to: {output}")
    return summary, run_config
