# URECA surface-EMG project

The current datasets, models, measured results, live features, and change log are tracked in [PROJECT_STATUS.md](PROJECT_STATUS.md).

This public repository contains source code, selected trained models, and summary results. Raw participant recordings and local runtime files are excluded. Model training and retrospective evaluation require the original recordings; the hand-only viewer can instead start with a fresh rest baseline using `vision_live.py --model models/vision_multisession_P3_P4_v1/cnn_plain.pt --baseline-mode fresh`.

The current seven-gesture, multi-posture model and local launch command are
documented in [COMPACT_MODEL.md](COMPACT_MODEL.md). Its unfamiliar-position test
uses a separate recording that was excluded from training.

Python software for an eight-channel surface-EMG controller and real-time hand-
gesture classification. The project currently supports raw transport inspection,
provisional frame decoding, labelled calibration recordings, traditional EMG
features, an LDA baseline, and rolling inference using synthetic/replayed/live input.

Synthetic output validates software behaviour only. It is not evidence of EMG
accuracy.

## Current hardware facts and hypotheses

Confirmed settings from the lab software are `COM10`, 921600 baud, eight channels,
and a configured 250 Hz internal sample rate. Direct serial measurement shows a
nominal 125 complete frames/s on the accessible stream. It contains 38-byte frames
starting with `55 66` and ending with `66 55`.

The decoder preserves the next two bytes as an unknown metadata word and treats the following
eight four-byte groups as big-endian signed channel counts. This interpretation and
the ADC-to-voltage scale are provisional. Live values are therefore stored as
`unscaled_adc_count`, never as volts. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Environment

The current Python 3.13 environment is compatible with the installed dependencies;
Python 3.11 or newer is accepted by the project.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The proprietary EMG executable and other serial monitors must be closed before any
Python command opens `COM10`.

## Electrode-free checks

Inspect completely raw bytes without assuming a protocol:

```powershell
python .\serial_probe.py --chunk-size 64
```

Measure the provisional decoded frame rate for 10 seconds:

```powershell
python .\frame_probe.py --seconds 10
```

The observed nominal transport rate is approximately 125 frames/s and 4750 bytes/s.
Floating/saturated disconnected channels are not valid EMG.

Run the automated tests:

```powershell
python -m unittest discover -s tests -v
```

## Synthetic calibration and LDA dry run

Create a labelled session without waiting for countdowns:

```powershell
python .\calibrate.py --source synthetic --fast --session demo_01
```

This creates:

```text
recordings/DRYRUN/demo_01/
  samples.csv
  events.csv
  metadata.json
```

Train the 32-feature LDA baseline with a grouped split:

```powershell
python .\train_model.py .\recordings\DRYRUN\demo_01 --output .\models\lda.joblib
```

Review sample integrity, saturation, and activity ratios:

```powershell
python .\review_session.py .\recordings\DRYRUN\demo_01
```

Exercise rolling 200 ms inference on a new synthetic signal:

```powershell
python .\realtime.py --model .\models\lda.joblib --source synthetic --gesture fist
```

Install the optional CNN dependency and train a session-held-out 1D CNN:

```powershell
python -m pip install -r requirements-cnn.txt
python .\train_cnn.py --train <training-session-folders> --test <test-session-folder>
```

CNN input has shape `(batch, 8 channels, 25 samples)`. Validation uses whole
segments for early stopping; the specified test session is evaluated only afterward.

Display live LDA predictions in a separate gesture-image window:

```powershell
python .\live_display.py --model .\models\s01_v2_seated_lda.joblib
```

The required image filenames and their mapping are documented in
`assets/gestures/README.md`. The window remains functional with a missing-image
placeholder until those local files are added.

## Local 3D hand web app

Install the browser renderer once, then start the app in demo mode to verify the
interface without using the receiver or electrodes:

```powershell
npm.cmd install
.\.venv\Scripts\python.exe .\webapp.py --demo
```

For live COM10 predictions, close every other serial program and run:

```powershell
.\.venv\Scripts\python.exe .\webapp.py
```

Open `http://127.0.0.1:8765` if the browser does not open automatically. Use **Fresh
calibration** after applying or moving electrodes: the page guides the baseline and
three interleaved trials of every gesture, trains a replacement LDA, and then starts
live inference. Use **Test saved model** only when electrode placement is unchanged.
The browser gesture set is `rest`, `splayed`, `fist`, `wrist_up`, and `wrist_down`.
For consistent calibration, `wrist_up` means wrist extension (bend the back of the
hand toward the top of the forearm) and `wrist_down` means wrist flexion (bend the
palm toward the inner forearm); keep the fingers relaxed during both.
The hand remains locked at rest until the deliberate wake sequence
`splayed, fist, splayed` is recognized. Rest predictions do not reset partial wake
progress, and the user has four seconds to begin the next pose. A saved model with the
previous gesture labels is shown as incompatible instead of being used silently.

This version maps discrete LDA labels to articulated 3D pose targets and smoothly
interpolates the finger joints between them. Continuous reconstruction of arbitrary
finger motion would require synchronized finger-joint ground truth and a regression
model, not only the current gesture labels.

The pipeline uses 25-sample windows, a 5-sample hop, MAV/WL/ZC/SSC for each channel,
`StandardScaler`, and multiclass LDA. Its evaluation split keeps each continuous
labelled segment wholly in training or testing so overlapping windows cannot leak
across the boundary.

## Vision-assisted ground truth recorder

The separate local recorder uses MediaPipe hand/pose landmarks to propose labels
for `rest`, `fist`, `splayed`, `peace`, `point`, `pinch`, and `thumbs_up`, then
synchronizes stable visual holds to the raw 125 Hz EMG sample index.
Elbow angle and forearm orientation are recorded as posture conditions rather than
gesture classes. A prompted visual calibration builds personal vertical- and
horizontal-forearm pose templates from 3D hand landmarks;
subsequent EMG collection is automatic and free-order. Camera frames are processed in the browser and
are not saved; landmarks, confidence, timing, and the accepted/rejected decision are
saved beside the EMG for audit. See [VISION_GROUND_TRUTH.md](VISION_GROUND_TRUTH.md).

Test the camera and workflow without electrodes or the receiver:

```powershell
npm.cmd install
.\.venv\Scripts\python.exe .\vision_ground_truth.py --source synthetic
```

For a real recording, close other serial programs, turn the transmitter on, and run:

```powershell
.\.venv\Scripts\python.exe .\vision_ground_truth.py --source serial --port COM10
```

Open `http://127.0.0.1:8767/vision.html` if needed. The vision labels are a
candidate ground truth source, not a validated measurement instrument: frame the
hand, wrist, and elbow, review rejected trials, and validate a subset manually
before using the sessions for model claims.

The frozen seven-gesture CNN can also be run on the local live inference page.
The held-out robustness benchmark, trial-level uncertainty intervals, and
representation experiments are documented in
[VISION_OFFLINE_IMPLEMENTATION.md](VISION_OFFLINE_IMPLEMENTATION.md).

The prospective seven-gesture electrode reapplication capture mode and frozen
evaluator are documented in [VISION_REAPPLICATION.md](VISION_REAPPLICATION.md).

## Baseline Benchmark v0

Measure the existing decoder on grouped held-out recordings without changing it:

```powershell
.\.venv\Scripts\python.exe .\benchmark.py
```

This uses `data/session_manifest.csv`, explicitly reports exclusions, and writes
raw/smoothed metrics, confusion matrices, per-window probabilities, and the exact
run configuration under `results/baseline_v0/`. The current-label selection has
one complete session, so the default refits the canonical LDA on training segments
and scores held-out segments. Saved models require disjoint test sessions.
See [BASELINE_BENCHMARK.md](BASELINE_BENCHMARK.md) for splitting, timing definitions,
cross-session commands, measured results, and limitations.

Run controlled acquisition-noise stress tests with clean training and a frozen
split/model:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-benchmark.txt
.\.venv\Scripts\python.exe .\benchmark.py --mode corruption
```

This defaults to v1.1 and writes `results/corruption_v1_1/`: noise amplitudes use
per-segment/channel mean-removed AC power, then noise is added to the original
raw counts before CAR/filtering. The 54-condition sweep includes five stochastic
repeats, per-channel severity curves, scale diagnostics, compressed predictions,
and plots. The held-out normalization baseline is corrupted too. The original
v0/v1 reports are preserved. See [Corruption Benchmark v1.1](BASELINE_BENCHMARK.md#corruption-benchmark-v11)
for definitions, measured results, and historical v1 reproduction commands.

Future calibration recordings can carry explicit posture, orientation, date,
electrode, and motion conditions using `calibrate.py --conditions-json
data/recording_conditions.example.json`. Edit the example to reflect the observed
conditions first. These fields are optional; detailed legacy positions remain
unknown. The benchmark supports repeated `--where FIELD=VALUE` filters; see the
benchmark documentation for the schema and grouping examples.

## Commands reserved for later hardware work

Dataset v2 recording and evaluation support is documented in
[Dataset v2 Protocol](DATASET_V2.md). It adds nine configurable classes, four static
posture blocks, independent long rest, stable trial IDs, and A/B/C application/day
metadata. The evaluator supports reference holdout, cross-posture, inclusive
training, reapplication, 0-3 repetition recalibration, and cross-day comparisons.
The existing decoder and browser gesture controller retain their current behavior.

Preview the 216-trial Session A schedule without opening the receiver:

```powershell
.\.venv\Scripts\python.exe .\calibrate.py --protocol dataset_v2 --conditions-json data/dataset_v2_session_A.example.json --plan
```

Edit the example conditions before collecting real sessions; the protocol guide
includes serial commands, separate manifest inventory, validation and evaluation.

`calibrate.py --source serial` and `realtime.py --source serial` exist so the same
interfaces can eventually process the controller. Do not treat their output as
scientific EMG data until channel encoding, scaling, signal quality, filtering, and
feature thresholds are validated.

For real sessions, always pass a standardized `--posture` identifier. Arm and
forearm orientation are experimental conditions, not incidental metadata.
