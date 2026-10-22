# EMG software architecture

## Scientific boundary

The software separates observations from hypotheses. The observed live capture has
38-byte frames with `55 66` and `66 55` markers. Treating the next eight four-byte
groups after an unknown two-byte field as big-endian signed channel counts is
provisional. ADC-to-volt scaling and the meaning of the unknown field remain
unconfirmed.

## Data path

All input implementations produce the same eight-channel `Sample` object:

```text
SyntheticSource ─┐
TextReplaySource ├─> Sample ─> recorder ─> windows ─> features ─> scaler/LDA
SerialFrameSource┘
```

This lets calibration, training, and inference be exercised without electrodes.
Raw session samples are retained in their source unit. Converting provisional ADC
counts to volts must happen only after the scale is verified.

## Calibration session

`calibrate.py` runs a sample-indexed state machine:

```text
baseline rest -> prepare -> gesture -> rest -> ... -> session end
```

Each output directory contains:

- `samples.csv`: all eight raw values plus the active phase and label;
- `events.csv`: exact sample indices for phase boundaries;
- `metadata.json`: source, unit, subject/session IDs, timing, and gesture set.

Trials are interleaved by gesture. A later user-interface revision can randomize the
order with a recorded seed once the experimental protocol is finalized.

## Baseline model

The controller is configured internally for 250 Hz, but clean direct measurements
show a nominal 125 complete frames/s on the accessible stream. Windows are therefore
25 samples (200 ms) with a 5-sample (40 ms) hop. Every channel yields
MAV, waveform length, zero crossings, and slope-sign changes, producing 32 features.
`train_model.py` uses a `StandardScaler` followed by multiclass LDA.

Overlapping windows from one continuous labelled segment share a group. Evaluation
splits whole groups, never individual windows, to prevent near-duplicate leakage.
The reported grouped score is separate from the final deployable model, which is
then fitted on all supplied windows.

## Implemented since the initial architecture notes

The current pipeline supports causal fourth-order 10–55 Hz filtering, optional
common-average referencing and rest-RMS normalization, segment transition guards,
probability smoothing in the display applications, and session-held-out LDA
evaluation. Browser calibration enables the reference and normalization options.
The original deferral list below has been updated to reflect implementation.
Baseline v0 measures these existing behaviors without changing them; see
[BASELINE_BENCHMARK.md](BASELINE_BENCHMARK.md) for exact implementation references,
reproducibility, and the offline/live reconnect-state discrepancy.

## Items still deferred until signal inspection

- ADC-to-volt scaling and channel mapping confirmation
- validated filter design and notch filtering
- noise-derived ZC/SSC thresholds
- saturation and contact-quality limits
- finalized gesture timing and validated transition guards
- probability rejection
- broader cross-session/electrode-reapplication evaluation

Synthetic results validate program flow only and are never evidence of EMG accuracy.
