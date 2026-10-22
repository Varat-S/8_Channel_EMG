# Baseline Benchmark v0

For the separate nine-class static-posture/reapplication collection and evaluation
workflow, see [Dataset v2 Protocol](DATASET_V2.md). V0 and corruption reports remain
historical controls with their original vocabulary and split.

This benchmark measures the existing decoder on unperturbed recordings. It does
not add noise or change preprocessing, thresholds, classifiers, serial decoding,
or the browser. "Clean" means no injected corruption; the recorded signals are
not certified noise-free.

From the `EMG` directory, with the dependencies in `requirements.txt` installed:

```powershell
.\.venv\Scripts\python.exe .\benchmark.py
```

The default uses `data/session_manifest.csv` and writes `results/baseline_v0/`.
To invoke the reusable module directly without installing the package:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
.\.venv\Scripts\python.exe -m emg.evaluation.benchmark --manifest data/session_manifest.csv --output results/baseline_v0
```

## Frozen implementation

The sources of truth are:

| Operation | Existing implementation reused |
|---|---|
| Reference/filter/normalization and windows | `emg.dataset.load_feature_dataset` |
| Causal fourth-order Butterworth | `emg.preprocessing.filter_recording` / `CausalBandpass` |
| MAV/WL/ZC/SSC | `emg.features.extract_time_domain_features` |
| StandardScaler and uniform-prior multiclass LDA | `emg.model._new_pipeline` |
| Saved model loading | `emg.model.load_model` |
| Probability smoothing formula | `PredictionController._serial_worker` in `webapp.py` |

The default is the current web calibration configuration: eight channels, 125 Hz,
common-average reference, causal 10–55 Hz filtering, initial rest-RMS normalization,
25-sample windows, 5-sample hop, zero ZC/SSC thresholds, and seven-window smoothing.
The loader removes 38 samples (304 ms at 125 Hz) from **each end** of a continuous
labelled segment, skips `prepare`, and does not bridge `stream_reconnect` boundaries.
Window metadata is reconstructed using its existing segment-boundary helper and
checked against the loader's labels/groups. Signal processing is not duplicated.

Architecture notes previously listed filtering, guards, temporal smoothing, and
cross-session evaluation as deferred. The current implementation already supplies
these. Baseline v0 follows the implementation, including LDA's uniform class priors.

Saved artifacts supply their own window/hop/fs/threshold/reference/filter and
normalization configuration. Missing optional keys retain the same legacy fallbacks
as live inference: no filter, reference `none`, and no baseline normalization.
Saved artifacts do not record transition guards or smoothing: benchmark v0 uses
38 guard samples and seven smoothing windows, and records this in `run_config.json`.

## Inventory and selection

The manifest inventories 21 session directories. Paths are relative to the
manifest's directory. Gesture sets use `|` delimiters and include the protocol's
rest class; metadata lists active gestures only. `group_id` identifies an independent
session for cross-session splitting. Completion is read from metadata; for older
sessions with no completion flag, a `session_end` event verifies completion, with
that inference recorded in notes. Unknown recording dates remain blank. Dates from
timestamped directory names are marked as inferred, with timezone unknown.

Only `web_20260926_192402` is enabled by default because it is complete, has valid
label timing, and matches `rest,splayed,fist,wrist_up,wrist_down`. Every excluded
session and its reason is printed and saved. Incomplete/unknown completion, invalid
timing, or incompatible labels are excluded. Missing files, stale manifest labels,
duplicate IDs/paths, non-finite channels, missing baselines, or non-contiguous sample
indices fail with diagnostics. Explicitly requesting an unusable session fails
rather than silently changing the requested experiment. Within-condition selections
must agree on subject/posture/source and all selections must agree on signal units
and sample rate. No labels are remapped or intersected.

Legacy sets require both explicit session selection and an exact `--gesture-set`
(or the classes of a supplied saved model). The generic subject identifier `WEB`
does not establish a biological subject's identity; all recorded metadata is saved
for interpreting comparisons.

To regenerate the inventory after adding recordings, review the resulting flags:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
.\.venv\Scripts\python.exe -m emg.evaluation.dataset --recordings recordings --output data/session_manifest.csv
```

This rebuilds the manifest from recorded facts; custom manifest annotations should
be kept separately before regeneration. No existing train/validation/test assignment
was found in the recording metadata. Baseline split assignments live in each run's
configuration.

## Split and model provenance

Default mode `within-session` uses the **first fold** of
`StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=7)`, matching current
`emg.model.train_and_evaluate`. A group is one continuous `(phase, gesture, trial)`
segment, split additionally at reconnect events, identified by session path and
segment number. Trial numbers repeat across gestures, so bare trial numbers are
not globally unique. Whole groups remain together; overlapping windows cannot be
on opposite sides. Both sides must contain every class. The scaler and LDA are
fitted only on training windows. The evaluation model is never refitted on all data.

`--mode cross-session` keeps whole sessions and manifest session groups disjoint.
Use `--train-sessions` and `--test-sessions` to make the partition explicit. With
`--sessions` alone, the last selected session in manifest order is held out and all
others train. Fewer than two independent compatible session groups produces a clear
error. Cross-session results use the same per-session rest calibration available
to live inference; this is not evaluation without baseline calibration.

A compatible legacy experiment is stored separately from the current-label baseline:

```powershell
.\.venv\Scripts\python.exe .\benchmark.py --mode cross-session --gesture-set rest,splayed,fist,thumbs_up,rock,pinch --train-sessions web_20260917_110358 --test-sessions web_20260926_185138 --output results/baseline_v0_legacy_cross_session
```

`--model` is allowed only in cross-session mode. The artifact must contain training
session provenance. Recorded training paths/names are checked against test sessions;
overlap fails. Relative Windows paths and relocated session-directory names are
recognized. An artifact fitted on all windows of a session cannot be used to score
that same session as held out. Default within-session mode therefore refits the
canonical LDA instead of using `web_latest_lda.joblib`, which trained on the only
current-label recording.

For example, a historical saved model can be evaluated on an unseen compatible
recording with:

```powershell
.\.venv\Scripts\python.exe .\benchmark.py --mode cross-session --model models/web_20260926_185138_lda.joblib --test-sessions web_20260917_110358 --output results/saved_model_holdout
```

Provenance is as reliable as the artifact's recorded session list; renamed or
unrecorded training data cannot be detected automatically.

## Metrics and smoothing scope

Macro F1 is the primary metric. Both raw and smoothed outputs include accuracy,
balanced accuracy, per-class precision/recall/F1/support, and a confusion matrix
with explicit model-class ordering. Classes are never hard-coded into probability
columns. `predicted_label`, `confidence`, and `prob_<class>` in `predictions.csv`
refer to the smoothed output; `raw_*` columns preserve raw LDA probabilities and
predictions. `end_sample` is exclusive, and timestamps derive from recorded
sample-index time, not independently measured hardware timestamps.

Smoothing uses the exact live calculation: `deque(maxlen=7)`, append probabilities,
`np.mean(history, axis=0)`, then `argmax`. Warmup averages available windows. History
is reset at every independent held-out segment or missing-window gap. Unlike a
continuous live run, evaluation omits prepare/guard/training windows, so carrying
history across those gaps would mix unrelated observations. This is **segment-local
smoothed classifier output**, not a full chronological replay. The wake sequence,
hold detector, confidence gate, and hand interpolation are outside these metrics.

Switching counts changes between adjacent predictions within each uninterrupted
held-out segment. The denominator is
`sum((sequence_windows - 1) * hop_size / sample_rate_hz)`. Boundaries never count as
switches. With no adjacent windows the rate is JSON `null`.

False-active fraction is `non_rest_predictions_on_rest / rest_windows`. Episodes
start when a rest-labelled window first becomes falsely active; an already active
first window counts as one left-censored episode. Changing between non-rest labels
during an episode does not add episodes. The approximate episodes/minute is
`episodes * 60 / (rest_windows * hop_size / sample_rate_hz)`. This treats each scored
window as one hop of exposure and excludes guards and omitted intervals. It is an
extrapolation from the available rest exposure, not observed minute-long activity.
Absent rest windows produce `null` rates. Both raw and smoothed rest metrics are saved.

## Generated artifacts and reproducibility

Each output directory contains:

- `metrics.json`: raw/smoothed classification, switching, and rest metrics.
- `per_class_metrics.csv`: both decoders' class metrics.
- `confusion_matrix.csv` and `confusion_matrix_raw.csv`.
- `confusion_matrix.png`: simple smoothed confusion plot using existing Pillow.
- `predictions.csv`: held-out provenance, truth, both predictions and probabilities.
- `run_config.json`: seed, parameters, sessions, exclusions, exact indices/groups,
  phase counts, split method, UTC timestamp, dependency versions, and input/source
  SHA-256 hashes. Git commit is recorded when available, otherwise `null`.
- `evaluation_model.joblib`: fitted evaluation checkpoint plus run configuration;
  this is deliberately separate from deployable live model artifacts.

Repeated runs with the same inputs, code, dependencies, and seed reproduce metrics
and prediction rows. Timestamp changes are expected. This folder currently has no
Git repository, so source hashes provide the implementation fingerprint.

## Measured Baseline v0 and limitations

Default run: `web_20260926_192402`, seed 7, **288 train / 225 test windows**.
Test groups contain one held-out segment for each active gesture and the initial
rest baseline. All later rest segments are in training. The initial baseline also
supplies the session's normalization scale, as in the canonical loader; thus this
rest score is not an independent test of fresh post-calibration rest. Filtering is
causal over the entire recording before grouped splitting, also matching training.

| Metric | Raw LDA | Segment-local smoothing |
|---|---:|---:|
| Accuracy | 0.728889 | 0.835556 |
| Balanced accuracy | 0.721905 | 0.829524 |
| Macro F1 | 0.706169 | 0.809704 |
| Switches / second | 5.568182 | 2.272727 |
| False-active rest windows / 105 | 27 | 16 |
| False-active rest fraction | 0.257143 | 0.152381 |
| Rest activation episodes | 10 | 5 |
| Approximate episodes / minute | 142.857143 | 71.428571 |

Rest exposure for these rate estimates is only **4.2 seconds** of window-cadence
exposure. The record includes two reconnect event rows at sample 9436. The loader
splits windows there, but **does not reset its filter state**; live inference resets
filter state after reconnect. Baseline v0 preserves and documents this discrepancy.
The resulting short tail cannot produce guarded windows. No noise removal beyond
the existing preprocessing is implied.

Smoothed confusion matrix (rows=true, columns=predicted):

```text
              fist  rest  splayed  wrist_down  wrist_up
fist            18     0        9           3         0
rest             0    89        3           0        13
splayed          0     6       21           0         3
wrist_down       0     0        0          30         0
wrist_up         0     0        0           0        30
```

No second compatible recording exists for the current five labels. The separate
legacy six-label cross-session run trains on `web_20260917_110358` and tests on
`web_20260926_185138` (630 train / 630 test windows). Raw/smoothed macro F1 is
0.308748/0.315961, balanced accuracy 0.334259/0.353704, and switches/sec
5.509182/1.794658. Those values describe a different gesture vocabulary and are
not a direct comparison against the default five-label within-session score.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Focused evaluation tests cover group/sample isolation, known classification values,
rest episodes, sequence boundaries, smoothing/warmup/gaps, manifest problems,
canonical feature equivalence, output reproducibility, cross-session holdout, and
saved-model provenance rejection. Synthetic tests check software behavior only.

## Historical Synthetic Corruption Benchmark v1

V1 adds controlled acquisition degradation around the same Baseline v0 evaluation,
without modifying the decoder. The raw recording files, serial protocol, CAR,
filter, thresholds, normalization, feature extraction, scaler, LDA, smoothing,
confidence handling, and CNN remain unchanged.

Install the optional plotting dependency and run from `EMG`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-benchmark.txt
.\.venv\Scripts\python.exe .\benchmark.py --mode corruption --corruption-version v1 --manifest data/session_manifest_v1.csv --corruption all --seed 7 --repeats 5 --output results/corruption_v1_reproduction
```

The original results remain in `results/corruption_v1/`. The command above writes
a separate reproduction directory; the original v0/v1 output paths are protected
against corruption-run overwrites. The default corruption version is now v1.1.
Matplotlib generates static plots and keeps its font cache in temporary storage.
Baseline v0 does not require Matplotlib. Use `--output` to select another directory.
The existing `--manifest`, `--sessions`, and exact `--gesture-set` options remain
available. Corruption v1 uses the grouped clean fit, so `--model` or explicit
train/test session assignments are rejected in this mode.

### Split, training and injection

The clean reference is prepared **once** using the original first fold of two-fold
`StratifiedGroupKFold`, split seed **7**, yielding the original 288 train / 225 test
windows for `web_20260926_192402`. `--seed` controls corruption only; it cannot change
this split. Scaler/LDA are fitted exactly once on clean training features, and the
same fitted model evaluates every condition. Fingerprints verify the model and
clean training feature arrays remain unchanged throughout the sweep. There is no
noise augmentation, retraining per condition, threshold tuning, or selected fix.

Injection occurs at the earliest available representation:

```text
recorded raw, unscaled ADC counts (CH0..CH7)
    -> synthetic corruption on full held-out labelled segments
    -> existing CAR
    -> existing causal filter
    -> existing baseline normalization
    -> existing features
    -> frozen clean-trained scaler/LDA
    -> existing segment-local smoothing and metrics
```

`SessionRecorder` stores raw samples, so this is not post-filter or feature noise.
The adapter caches original rows and raw arrays, transforms a copy of every full
held-out segment including the 38-sample guards at either end, and feeds temporary
CSV overlays into the **unchanged canonical loader**. It does not write to the
recordings. Prepare and training segments remain raw-clean. Reconnect boundaries,
labels, trial IDs, window locations, and test groups are unchanged; the split is
never regenerated. Only held-out feature rows from the loader's overlay output
are used, and training feature rows remain the clean originals.

Filtering still processes each whole recording causally without additional resets,
as in v0. A corrupted earlier held-out segment can affect the filter state entering
later intervals. Reprocessing can also change temporary training-row features
through normalization, but these rows are discarded and never used for fitting.
There are no new filter initialization semantics at segment boundaries.

The normalization baseline is treated according to its original split assignment.
If held out, its **entire raw interval** is corrupted and the unchanged loader
recomputes filtered baseline RMS from it. If assigned to training, its raw interval
remains clean. The default run holds out the initial baseline, so each default
condition affects both that baseline and the four held-out active segments. The
fitted scaler/LDA keep their clean training statistics. Thus v1 tests degradation
of acquisition **including its session normalization baseline**, not noise added
after a fixed clean normalization scale. V0's shared calibration/evaluation rest
caveat and short 4.2-second window-cadence rest exposure still apply.

### Definitions and configuration

Historical sweep constants are in `emg.evaluation.corruptions.LEGACY_CORRUPTION_CONFIG` and the
resolved conditions are saved in `run_config.json`. The default matrix has
**25 conditions / 89 individual evaluations**, counting the clean reference once.

| Family | Default levels | Definition before CAR |
|---|---|---|
| Clean | none | Exact prepared v0 result, with no added corruption |
| Broadband Gaussian | 20, 10, 5, 0 dB | Independent samples/channels, variance `P[ch] / 10**(SNR/10)` |
| 50 Hz | 0.10, 0.25, 0.50, 1.00 | Sine peak amplitude `A[ch] = multiplier * sqrt(P[ch])` |
| Channel dropout | CH1..CH8 | Zero the selected raw channel |
| Single-channel Gaussian | CH1..CH8 at 0 dB | Gaussian noise on just the selected raw channel |

`P[ch] = mean(raw_samples[:, ch]**2)` is computed separately for each channel and
full evaluation segment, **including DC**. It is not demeaned signal variance,
useful EMG-band power, or post-filter SNR. The raw counts in this recording have a
substantial DC offset, so these nominal SNR/RMS values can create noise much larger
than the useful AC variation. In particular, 20 dB cannot be interpreted as mild
contamination of filtered EMG. `signal_audit.json` reports both total and demeaned
raw power and actual corruption RMS per segment/channel to make this inspectable.

Gaussian noise is zero-mean in distribution; finite-realization mean and SNR can
fluctuate. Its variance is not rescaled to force an exact finite-sample SNR. Zero
or underflowed raw channel power produces zero added Gaussian/sine noise, with no
absolute noise floor or division by signal power. Nonzero tiny power produces
proportionally tiny noise. Input arrays are never mutated.

At 125 Hz, 50 Hz is below Nyquist. A sine's injected RMS is approximately
`A/sqrt(2)`; the listed multiplier sets its **peak amplitude**, matching the requested
formula. Independent uniform phases are generated per channel and segment. No
notch or other filter change is applied. Channel reports use **CH1..CH8**;
CSV columns and corruption API indices use **CH0..CH7 / 0..7** respectively.

The API accepts optional multi-channel dropout tuples, and configurable SNR for
single-channel Gaussian noise. All 28 channel pairs can be requested explicitly:

```powershell
.\.venv\Scripts\python.exe .\benchmark.py --mode corruption --corruption-version v1 --corruption channel-dropout --include-pair-dropout --output results/corruption_v1_pairs
```

Pairs are not included in the default matrix. Select an individual family with
`--corruption broadband`, `powerline`, `channel-dropout`, or `channel-noise`.
`--corruption none` produces just the clean reference. Every selection includes clean.

### Randomness and repeated measurements

The default stochastic repeats use seeds **7, 8, 9, 10, 11**. Dropout and clean
run once. Each repeat creates independent segment generators through NumPy
`SeedSequence([repeat_seed, four uint32 words of SHA256(session_id)[:16], segment_number])`.
This avoids Python's randomized hash and dependencies on which other conditions
were selected. The same segment substreams are matched across severity levels;
phases/noise realizations are paired to compare severity with less random variation.
There is no global random state. All seeds and their derivation are recorded.

Summary rows report means and **sample SD (`ddof=1`)** across stochastic repeats,
with SD zero for a single run or deterministic condition. Delta metrics subtract
the same clean reference. Predicted labels/probabilities are never averaged across
realizations. Compressed prediction files use fixed gzip header timestamps so
identical runs also reproduce their bytes. Repeat SD describes realization spread,
not subject/session uncertainty or a confidence interval.

### Outputs

- `summary.csv` / `summary.json`: one aggregate row per condition, raw/smoothed
  accuracy, balanced accuracy, macro F1, switching, rest false activations,
  episode counts, SDs, and deltas from clean. JSON also indexes individual runs
  and records worst conditions/channels and gesture recall sensitivity.
- `run_config.json`: frozen clean split/provenance, corruption definitions,
  seeds, normalization/injection semantics, dependency versions, and fingerprints.
- `per_class_summary.csv`: precision, recall, F1, SD, and change from clean for
  both decoders. `recall_changes_raw.csv` and `recall_changes_smoothed.csv` provide
  wide condition-by-gesture recall tables with deltas and SDs.
- `<family>/<condition>[_seedN]/`: individual metrics, numerical confusion
  matrices for both decoders, class metrics, and signal audits.
- `predictions/<condition>[_seedN].csv.gz`: v0's shared per-window schema, stored
  once per realization, with raw/smoothed probabilities and original provenance.
- `plots/`: broadband and 50 Hz F1 curves (raw and smoothed, with SD), channel
  dropout/noise bars, and five selected smoothed confusion images.
- `evaluation_model.joblib`: one frozen evaluation checkpoint for the whole sweep.

Worst-channel plots are selected by the lowest **aggregate smoothed macro F1**.
The selected confusion PNGs show seed 7 (the first realization) for clean,
broadband 0 dB, 50 Hz 1.00, worst dropout, and worst noisy channel. They are not
confusion matrices averaged across realizations. Numerical matrices for every
realization remain available. Predictions retain original segment boundaries and
the exact same smoothing resets and rest/switch metric definitions as v0.

### Measured results and interpretation

The full default sweep confirms clean smoothed macro F1 **0.8097042606516291**
and balanced accuracy **0.8295238095238094**, unchanged from v0.

| Condition | Smoothed macro F1 mean ± SD | Delta F1 | Balanced accuracy | False-active rest fraction | Switches/sec |
|---|---:|---:|---:|---:|---:|
| Clean | 0.8097 ± 0.0000 | 0.0000 | 0.8295 | 0.1524 | 2.2727 |
| Broadband 20 dB | 0.1581 ± 0.0368 | -0.6516 | 0.2072 | 0.0838 | 0.6591 |
| Broadband 10 dB | 0.1510 ± 0.0391 | -0.6588 | 0.2011 | 0.0876 | 0.8182 |
| Broadband 5 dB | 0.1525 ± 0.0407 | -0.6572 | 0.2025 | 0.0876 | 0.7273 |
| Broadband 0 dB | 0.1525 ± 0.0407 | -0.6572 | 0.2025 | 0.0876 | 0.7273 |
| 50 Hz 0.10 | 0.0536 ± 0.0147 | -0.7561 | 0.2000 | 1.0000 | 0.0000 |
| 50 Hz 0.25 | 0.0536 ± 0.0147 | -0.7561 | 0.2000 | 1.0000 | 0.0000 |
| 50 Hz 0.50 | 0.0536 ± 0.0147 | -0.7561 | 0.2000 | 1.0000 | 0.0000 |
| 50 Hz 1.00 | 0.0536 ± 0.0147 | -0.7561 | 0.2000 | 1.0000 | 0.0000 |

50 Hz interference is the worst tested family, with all four severities tied under
this setup. These are aggregate metric ties, not equal corrupted samples. Matched
phases, strong raw-DC-relative contamination, and corrupted baseline normalization
can produce a flat classification response; this experiment does not locate the
failure threshold below 0.10× raw RMS. Broadband already causes a large loss at
20 dB and has nearly flat scores over the requested sweep. Lower switching or a
lower false-active rest fraction under Gaussian noise does not imply robust
classification: balanced accuracy is near 0.2 and predictions can become stuck.

| Channel | Dropout F1 | Dropout delta | 0 dB single-channel-noise F1 mean ± SD | Noise delta |
|---|---:|---:|---:|---:|
| CH1 | 0.5955 | -0.2142 | 0.1314 ± 0.0187 | -0.6783 |
| CH2 | 0.8572 | +0.0475 | 0.1234 ± 0.0056 | -0.6863 |
| CH3 | 0.5580 | -0.2517 | 0.1250 ± 0.0094 | -0.6847 |
| CH4 | 0.7279 | -0.0818 | 0.1266 ± 0.0129 | -0.6831 |
| CH5 | 0.5786 | -0.2311 | 0.1250 ± 0.0094 | -0.6847 |
| CH6 | 0.6936 | -0.1161 | 0.1460 ± 0.0331 | -0.6637 |
| CH7 | 0.6906 | -0.1191 | 0.1322 ± 0.0150 | -0.6775 |
| CH8 | 0.5603 | -0.2494 | 0.1338 ± 0.0204 | -0.6759 |

Worst dropout is **CH3**, worst noisy channel **CH2**. CH2 dropout improves this
one held-out split's F1, which is not evidence to remove it from the decoder.
By equal-weight mean recall change across the 24 non-clean conditions,
**wrist_down** is most affected (mean recall change -0.7642); its recall falls
from 1.0 to 0.0 with CH3 dropout. Class ordering and full condition-specific
recall/confusion data are available in the generated tables.

These synthetic corruptions are controlled stress tests and are not claimed to
perfectly reproduce real electrode displacement, motion artefact, or skin-electrode
impedance changes. The study has one compatible recording and repeats of artificial
noise, so channel rankings and corruption-family comparisons are conditional on
this split, power definition, and normalization setup. No robustness intervention
has been selected or implemented.

Tests verify channel-specific target SNR, the 50 Hz spectral component at 125 Hz,
dropout and single-channel isolation, near-zero-power behavior, non-mutation,
segment substreams, repeated-evaluation reproducibility, unchanged training/model,
and exact agreement with the stored full-precision real Baseline v0 metrics,
including a no-op overlay through the canonical loader.

## Corruption Benchmark v1.1

Corruption Benchmark v1 based corruption amplitude on raw RMS/power, which included
a large ADC DC offset. This caused synthetic corruption amplitudes to substantially
exceed the useful AC EMG variation.

Corruption magnitude is now determined from mean-removed AC EMG power/RMS, while
corruption is still injected into raw ADC samples before decoder preprocessing.
For each full evaluation segment and channel, `x_ac = x_raw - mean(x_raw)`,
`P_AC = mean(x_ac**2)`, and `AC_RMS = sqrt(P_AC)`. Centering is used only to estimate
the amplitude; the decoder receives original raw counts plus corruption. The
implementation subtracts an initial reference before estimating the mean to avoid
unnecessary cancellation with large offsets. CAR, causal 10-55 Hz filtering,
normalization, windowing, features, scaler/LDA, smoothing, and wake logic are unchanged.

Synthetic corruption affects the entire held-out acquisition, including the
baseline samples used for normalization. Here, "held-out acquisition" means the
five full segments in the frozen test groups, including their guards: initial
baseline and one segment of each active gesture. Prepare/training raw segments
remain clean. The canonical loader recomputes baseline normalization from the
corrupted baseline; fitting always uses the original clean training features.
The same continuous filter state and segment-local smoothing semantics apply as v0.

### Run and configuration

From `EMG`, after installing `requirements-benchmark.txt`:

```powershell
.\.venv\Scripts\python.exe .\benchmark.py --mode corruption
```

The default output is `results/corruption_v1_1/`. Version selection is explicit
through `--corruption-version v1.1` or historical `v1`; v1 defaults to a separate
`results/corruption_v1_reproduction/` directory. `data/session_manifest_v1.csv`
archives the original manifest for historical provenance. Version v1 was checked
against the original report: all 25 aggregate rows and 89 compressed prediction
files reproduced exactly, while the original artifacts remained untouched.

| Family | v1.1 default levels | Amplitude before CAR |
|---|---|---|
| Broadband Gaussian | 30, 25, 20, 15, 10, 5, 0 dB | Variance `P_AC[ch] / 10**(SNR/10)` |
| 50 Hz | 0.01, 0.025, 0.05, 0.10, 0.25, 0.50 | Sine **peak** amplitude `A[ch] = multiplier * AC_RMS[ch]` |
| Dropout | CH1..CH8 | Same zero-raw-channel experiment as v1 |
| Single-channel Gaussian | CH1..CH8 at 20, 10, 5, 0 dB | Same AC variance formula on the selected channel only |

The default matrix has **54 conditions / 234 evaluations**: one clean reference,
eight deterministic dropouts, and 45 stochastic conditions with five repeats,
seeds **7-11**. The split is still seed 7 with 288 train / 225 test windows. Seed
substreams and matched realizations across severities are unchanged. Gaussian
targets describe expected variance; finite realization SNR is not forcibly rescaled.
A constant channel has zero AC power and therefore zero added noise, even with
nonzero DC. No absolute noise floor is introduced. A sine's injected RMS is
approximately its peak amplitude divided by `sqrt(2)`.

Add 15 dB or other finite levels explicitly:

```powershell
.\.venv\Scripts\python.exe .\benchmark.py --mode corruption --corruption channel-noise --channel-noise-snr 20 15 10 5 0 --output results/corruption_v1_1_channel_finer
```

Single-family selections always include the clean reference. `--corruption none`
reproduces the stored full-precision v0 metrics and its decompressed prediction
CSV byte for byte. Rerunning the complete v1.1 default matrix reproduced all 54
aggregate rows and all 234 compressed prediction files exactly. Timestamps in
run configuration are expected to change.

### Raw versus AC scale diagnostics

`signal_scale_summary.csv` has one row per session/segment/channel, with boundaries,
phase, gesture, held-out status, `raw_mean`, `raw_rms`, `ac_rms`, and
`raw_to_ac_rms_ratio`. `signal_scale_statistics.json` supplies all-segment and
held-out summaries. Zero-AC ratios would be null, never infinity; none occur here.

| Diagnostic | All segments (304 rows) | Held-out segments (40 rows) |
|---|---:|---:|
| Median raw mean (counts) | 222093.27 | 233672.66 |
| Median raw RMS (counts) | 222123.26 | 233699.92 |
| Median AC RMS (counts) | 2992.40 | 2216.03 |
| Median raw/AC RMS ratio | 66.39 | 102.35 |
| Minimum raw/AC RMS ratio | 2.90 | 31.32 |
| Maximum raw/AC RMS ratio | 449.91 | 374.61 |

Each median is computed separately. The median ratio describes about 102 times
too much amplitude under v1's raw-RMS scaling on the held-out segments; equivalent
power overstatement is the squared ratio. Mean-removed full-segment AC includes
drift and existing artefacts and is **not specifically 10-55 Hz useful-band power**.
These are pre-CAR, pre-filter scales, not measured post-filter SNR.

### Corrected measurements

All values below are segment-local smoothed output. F1 variability is sample SD
over the five synthetic realizations, not independent-session uncertainty.
Clean macro F1 remains **0.8097042606516291** and balanced accuracy remains
**0.8295238095238094**.

| Condition | Macro F1 mean +/- SD | Delta F1 | Balanced accuracy | False-active rest fraction | Switches/sec |
|---|---:|---:|---:|---:|---:|
| Clean | 0.8097 +/- 0.0000 | 0.0000 | 0.8295 | 0.1524 | 2.2727 |
| Broadband 30 dB | 0.7641 +/- 0.0668 | -0.0456 | 0.7524 | 0.1314 | 1.6364 |
| Broadband 25 dB | 0.6123 +/- 0.0451 | -0.1974 | 0.6133 | 0.1200 | 1.5909 |
| Broadband 20 dB | 0.4140 +/- 0.0689 | -0.3957 | 0.4575 | 0.0857 | 1.2727 |
| Broadband 15 dB | 0.3347 +/- 0.0430 | -0.4750 | 0.3966 | 0.0571 | 0.8182 |
| Broadband 10 dB | 0.2915 +/- 0.0548 | -0.5182 | 0.3358 | 0.0876 | 1.0455 |
| Broadband 5 dB | 0.2017 +/- 0.0644 | -0.6081 | 0.2434 | 0.0895 | 0.7500 |
| Broadband 0 dB | 0.1534 +/- 0.0228 | -0.6563 | 0.2021 | 0.1162 | 0.7273 |
| 50 Hz 0.01 x AC RMS | 0.7631 +/- 0.0408 | -0.0466 | 0.7846 | 0.2438 | 2.0682 |
| 50 Hz 0.025 x AC RMS | 0.7127 +/- 0.0422 | -0.0970 | 0.7242 | 0.2857 | 2.0000 |
| 50 Hz 0.05 x AC RMS | 0.5159 +/- 0.0429 | -0.2938 | 0.5524 | 0.5448 | 2.8864 |
| 50 Hz 0.10 x AC RMS | 0.2575 +/- 0.0656 | -0.5522 | 0.3438 | 0.8610 | 2.0455 |
| 50 Hz 0.25 x AC RMS | 0.0860 +/- 0.0394 | -0.7237 | 0.2067 | 1.0000 | 0.9318 |
| 50 Hz 0.50 x AC RMS | 0.0573 +/- 0.0155 | -0.7524 | 0.2040 | 1.0000 | 0.6591 |

The corrected curves locate progressive degradation. Broadband loses about 0.20
F1 by 25 dB; 50 Hz loses about 0.29 by 0.05 x AC RMS. Lower switches or rest
activation under strong Gaussian corruption can reflect predictions stuck on a
class and should be interpreted alongside F1 and balanced accuracy.

Single-channel noise mean smoothed macro F1:

| Channel | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---:|---:|---:|---:|
| CH1 | 0.6144 | 0.5375 | 0.3917 | 0.3456 |
| CH2 | 0.8318 | 0.7956 | 0.6929 | 0.5032 |
| CH3 | 0.7997 | 0.7249 | 0.7464 | 0.6212 |
| CH4 | 0.7878 | 0.6851 | 0.6704 | 0.6176 |
| CH5 | 0.7887 | 0.6963 | 0.6279 | 0.4371 |
| CH6 | 0.6326 | 0.3182 | 0.2745 | 0.2781 |
| CH7 | 0.7577 | 0.5020 | 0.3591 | 0.3341 |
| CH8 | 0.7743 | 0.6904 | 0.5721 | 0.4421 |

Clear losses already occur at the mildest tested single-channel severity,
**20 dB**, for CH1 (loss 0.1953) and CH6 (loss 0.1771). Worst channel is **CH1 at
20 dB**, then **CH6 at 10, 5, and 0 dB**. The actual onset may lie above 20 dB;
this sweep cannot locate that untested threshold. CAR distributes target-channel
noise to other channels, but this experiment also includes changed baseline
normalization and frozen-classifier sensitivity; it does not isolate CAR's causal
contribution. The lowest score anywhere in the channel sweep is CH6 at 5 dB.

Dropout rows are identical to historical v1, including means/deltas and all other
metrics. CH1..CH8 F1 remains respectively **0.5955, 0.8572, 0.5580, 0.7279, 0.5786,
0.6936, 0.6906, 0.5603**. CH3 is still worst. An improvement with CH2 dropout on
this single split is insufficient to justify removing that channel.

### Structured class errors

The numerical confusion matrices and per-class recall/F1 changes are retained
for every realization. The following are **mean error-window counts** across
repeats (support: fist/splayed 30 each, rest 105); clean has one realization.

| Condition | fist -> splayed | rest -> wrist_up | splayed -> rest | splayed -> wrist_up |
|---|---:|---:|---:|---:|
| Clean | 9.0 | 13.0 | 6.0 | 3.0 |
| Broadband 30 dB | 1.6 | 12.2 | 8.8 | 4.8 |
| Broadband 25 dB | 0.0 | 12.4 | 9.2 | 10.6 |
| 50 Hz 0.025 x AC RMS | 6.4 | 25.8 | 8.2 | 4.8 |
| 50 Hz 0.05 x AC RMS | 3.0 | 51.6 | 8.8 | 7.0 |
| CH1 noise 20 dB | 11.2 | 8.8 | 7.6 | 1.8 |
| CH6 noise 20 dB | 0.0 | 13.2 | 16.6 | 12.2 |

50 Hz clearly amplifies rest -> wrist_up; CH6 noise amplifies splayed -> rest/
wrist_up; CH1 noise increases fist -> splayed and introduces other fist errors.
Fist recall with CH1 noise at 20 dB is zero (clean 0.60); splayed recall with CH6
noise at 20 dB is 0.04 (clean 0.70). Fewer examples of a particular confusion can
mean errors have moved elsewhere: broadband 25 dB reduces fist -> splayed while
fist recall falls to 0.22. At broadband 30 dB wrist_down recall is already 0.787
from clean 1.0, while wrist_up remains 0.98. All five classes' full measurements
are in `per_class_summary.csv` and `recall_changes_smoothed.csv`.

### Artifacts and validation

V1.1 retains v1's raw and smoothed metrics, SDs, and deltas for F1, balanced
accuracy, false-active fraction and switching, plus per-class metrics, complete
predictions, per-realization signal audits and numerical confusion matrices.
The new scale files sit alongside `summary.csv`, `summary.json`, `run_config.json`,
and the single frozen evaluation checkpoint. `plots/` contains both F1 curves
(raw/smoothed mean +/- sample SD), dropout bars, lowest-SNR channel-noise bars,
the channel-by-SNR heatmap, and five selected confusion images. The confusion
images are first-realization matrices, selected by aggregate F1 where applicable.

The complete test suite passes **65 tests**. New coverage verifies AC offset
invariance, channel-wise measured AC SNR at all seven broadband levels, 50 Hz
peak/RMS scaling with large DC, constant-channel behavior, targeted corruption,
custom sweeps, metadata validation/filtering/grouping, legacy manifests, and a
synthetic recording with explicit conditions. Integration checks preserve full-
precision clean metrics, byte-identical clean predictions, unchanged fitting and
deterministic repeats. The full real-recording sweep was run twice with identical
54 aggregate rows and 234 compressed prediction files; all original v0/v1 files
were checked unchanged.

These curves show decoder sensitivity under corrected relative amplitudes in this
one acquisition, including its corrupted rest calibration. Synthetic stress tests
are not equivalent to electrode displacement, arm-posture shifts, or real motion
artefact. No independent current-label session or physical noise measurement is
available to establish real-world operating limits. Real multi-posture/session
data will be collected separately. No robustness solution has been implemented
or chosen.

### Future recording conditions and grouping

`data/session_manifest.schema.json` documents normalized manifest rows. Existing
`session_id`, `subject_id`, coarse `posture`, `group_id`, `gesture_set`, source,
units, and sample-rate fields remain. The optional condition fields are:

- `recording_date` (ISO YYYY-MM-DD), `electrode_placement_id`, `electrode_reapplied`.
- `arm_posture`, `shoulder_position`, `elbow_position`, `forearm_orientation`.
- `dynamic_arm_motion`, `calibration_session_id`, `notes`.

Boolean values are JSON true/false (CSV `true`/`false`) or unknown/null. Controlled
arm values are `reference`, `elbow_flexed`, `elbow_extended`, `arm_abducted`,
`arm_near_torso`, `custom`, `stationary_unspecified`; forearm values are `neutral`,
`pronated`, `supinated`, `custom`. Shoulder/elbow positions are explicit text so
future protocols can name measured positions without enforcing invented angles.
Legacy manifests with only the original columns remain valid. Missing detailed
positions remain unknown/null; no elbow, shoulder or forearm angles were inferred.
The current coarse `posture` identifiers retain their existing recorded meaning.

An illustrative future session's metadata can contain:

```json
{
  "session_id": "reference_day1_trial1",
  "subject_id": "S01",
  "recording_date": "2026-10-06",
  "electrode_placement_id": "placement_A",
  "electrode_reapplied": false,
  "arm_posture": "reference",
  "shoulder_position": "arm_near_torso",
  "elbow_position": "90_deg_flexion",
  "forearm_orientation": "neutral",
  "dynamic_arm_motion": false,
  "calibration_session_id": "reference_day1_calibration"
}
```

This example describes a future explicitly observed condition, not an existing
recording. `calibrate.py --conditions-json` accepts only the optional fields
listed above, so `session_id`/`subject_id` still come from `--session`/`--subject`.
Edit `data/recording_conditions.example.json` before recording; it defaults its
date/calibration link to null and marks its conditions as illustrative.

```powershell
.\.venv\Scripts\python.exe .\calibrate.py --source serial --subject S01 --session reference_day1_trial1 --posture reference --conditions-json data/recording_conditions.example.json
```

This records condition metadata without changing acquisition or the decoder.
Regenerate and review the manifest using the inventory command above. Nonempty
optional annotations in a manifest override recorded optional condition fields;
blank fields fall back to recorded metadata. Preserve manual annotations before
regenerating the manifest. Current recorded notes/date provenance are retained.

Repeated `--where FIELD=VALUE` arguments apply exact, case-insensitive AND filters,
with boolean validation. Missing values do not match known conditions. For example,
after collecting suitable recordings:

```powershell
.\.venv\Scripts\python.exe .\benchmark.py --where arm_posture=reference --where electrode_reapplied=false --output results/reference_condition
```

An explicit session selector remains available through `--sessions` or
`--where session_id=...`. `emg.evaluation.dataset.group_sessions_by_metadata`
groups selected `Session` objects by specified fields, including recording date,
electrode placement and posture. The existing within-condition compatibility
check now checks known detailed posture/electrode conditions too. Cross-session
mode can still use explicit train/test IDs when suitable recordings exist; no new
posture experiment, cross-posture adaptation, or training method was added.
