# Dataset v2 Protocol

Dataset v2 collects real recordings for larger-vocabulary, static-posture,
electrode-reapplication and cross-day comparisons. The decoder is unchanged:
8 channels at nominal 125 Hz, CAR, causal 10-55 Hz filtering, short-rest RMS
normalization, 25-sample windows / 5-sample hop, MAV/WL/ZC/SSC, StandardScaler,
uniform-prior LDA and seven-window probability smoothing. No robustness algorithm,
augmentation, confidence rejection, or CNN change is included.

Older folders whose names contain `v2` are not automatically Dataset v2 recordings.
The new protocol is explicitly identified by `metadata.json`'s
`protocol = dataset_v2` and version 2, with registered trial/condition metadata.
Legacy acquisition and evaluation remain supported; no labels or physical
conditions are inferred or remapped.

## Gesture and posture definitions

The default vocabulary has nine classes:

| Label | Prompt / consistent interpretation |
|---|---|
| `rest` | Relax wrist and fingers; explicit rest trials are part of the vocabulary. |
| `fist` | Close the fingers into a comfortable fist. |
| `splayed` | Extend and spread the fingers. |
| `wrist_up` | Wrist extension: back of hand moves toward the top of the forearm; fingers relaxed. |
| `wrist_down` | Wrist flexion: palm moves toward the inner forearm; fingers relaxed. |
| `radial_deviation` | Bend the wrist toward the thumb side without rotating the forearm. |
| `ulnar_deviation` | Bend the wrist toward the little-finger side without rotating the forearm. |
| `pinch` | Bring thumb and index fingertip together; keep the other fingers in a consistent relaxed position. |
| `point` | Extend the index finger, curl the other fingers; keep thumb position consistent. |

Use comfortable, repeatable poses and keep definitions consistent across sessions.
The CLI's `--gestures` accepts a custom exact vocabulary, including `rest` and at
least one active class. Supply that same vocabulary to the evaluator. This does
not change the browser's existing five-label gesture controller.

| Static posture | Interpretation |
|---|---|
| `reference` | The user's normal calibration posture; describe it in notes. |
| `elbow_flexed` | Elbow approximately 90 degrees. |
| `elbow_extended` | Arm substantially straighter than reference. |
| `arm_abducted` | Upper arm moved away from the torso. |

Keep forearm orientation `neutral` by default. `pronated` and `supinated` are
supported metadata values for a deliberate orientation experiment, not mandatory
extra blocks. Exact shoulder/elbow positions are optional descriptive strings
(include units for measurements). No exact joint-angle measurement is required.
All Dataset v2 blocks use `dynamic_arm_motion = false`; do not continuously move
the shoulder/elbow during a gesture. There is no factorial posture expansion.

## Sessions and placement IDs

| Role | Default protocol | Placement / date meaning |
|---|---|---|
| A | 9 gestures x 4 postures x 6 repetitions = **216 trials** | One electrode application throughout all blocks. |
| B | Reference posture, 9 gestures x 6 repetitions = **54 trials** | Remove/reapply electrodes; assign a new placement ID. Prefer the same day as A to isolate reapplication. |
| C | Reference posture, same gestures x 6 repetitions = **54 trials** | Different day from A, same nominal arrangement, new placement/application ID. Optional. |

Example IDs identify the current arrangement/application as **P2**, the reapplied
application as **P3**, and the later day's application as **P4**. The current ring
was shifted relative to the older recording; that is a distinct placement, not
a quantified displacement. Existing legacy placement fields remain unknown;
they were not automatically labeled P1. Record observed placement notes and use
unique IDs for actual applications.

Session A must keep the same application through every posture. Moving the arm
does not create a new placement ID; removing/reapplying electrodes does.
The evaluator requires distinct A/B/C session and application IDs, matching
subject/source/units/sample rate, and different A/C dates. If B is collected on
another day, the reapplication comparison also includes a day change; the run
configuration preserves dates so this confounding is visible.

## Recording workflow

From the `EMG` directory, inspect/edit these examples before collecting:

- `data/dataset_v2_session_A.example.json`
- `data/dataset_v2_session_B.example.json`
- `data/dataset_v2_session_C.example.json`

Unlike the legacy `--conditions-json` interface, the explicit Dataset v2 protocol
accepts session/subject identity and a gesture-set preset name in the JSON. CLI
identity values must agree with the JSON if both are supplied. The preset name
`dataset_v2_9class` identifies the nine default gestures; the actual `gestures` list
in metadata is always the authoritative class vocabulary.

Preview Session A's schedule without opening a device:

```powershell
.\.venv\Scripts\python.exe .\calibrate.py --protocol dataset_v2 --source serial --conditions-json data/dataset_v2_session_A.example.json --plan
```

Close other serial programs and record each session when ready:

```powershell
.\.venv\Scripts\python.exe .\calibrate.py --protocol dataset_v2 --source serial --session-role A --conditions-json data/dataset_v2_session_A.example.json
.\.venv\Scripts\python.exe .\calibrate.py --protocol dataset_v2 --source serial --session-role B --conditions-json data/dataset_v2_session_B.example.json
.\.venv\Scripts\python.exe .\calibrate.py --protocol dataset_v2 --source serial --session-role C --conditions-json data/dataset_v2_session_C.example.json
```

Without a conditions file, equivalent Session A identity arguments are:

```powershell
.\.venv\Scripts\python.exe .\calibrate.py --protocol dataset_v2 --source serial --session-id v2_session_A --subject-id S01 --placement-id P2
```

The terminal announces posture changes, then prompts each gesture/repetition.
Acquisition continues during the 15-second posture-change interval and preparation
countdowns, saving those samples as `prepare`; this drains the stream during
transitions. Settle before the countdown ends. There is no blocking input prompt
that leaves stale serial frames queued during a posture change.

Default timing:

1. Settle in reference posture (15 seconds).
2. Short normalization baseline (5 seconds).
3. Independent `rest_pre` (60 seconds, reference posture).
4. Gesture blocks: 3-second preparation, 3-second gesture, 2-second inter-trial rest.
5. Return to reference posture if needed; independent `rest_post` (60 seconds).

This is about 32 minutes for A and 9.5 minutes for B/C with no stream pauses.
Use `--independent-rest-seconds 45` for 45-second independent rest; both segments
share that setting. Other duration options are configurable. Actual sample
boundaries, sample-derived durations and measured wall durations are saved.
Sample-derived timing remains nominal 125 Hz, not independently measured hardware
time. Stream reconnect warnings are saved; pause time can increase wall duration.

`rest_pre` and `rest_post` are distinct phases/IDs. They never become the short
normalization baseline automatically. Both are collected in reference posture;
they do **not** measure long-rest stability separately in altered postures.

Gesture order is independently shuffled in each repetition block, with seed 7
by default. All gestures occur before that block repeats. `--seed` changes the
deterministic recording schedule. Reference is always first; the default remaining
postures are flexed, extended, abducted. `--randomize-postures` randomizes the
remaining order. `--postures reference arm_abducted elbow_extended elbow_flexed`
can set another order. Planned and actual orders are both saved.

Conditions can change per block without changing session/application identity:

```json
{
  "session_id": "v2_session_A",
  "subject_id": "S01",
  "electrode_placement_id": "P2",
  "forearm_orientation": "neutral",
  "dynamic_arm_motion": false,
  "block_conditions": {
    "reference": {"notes": "Describe normal calibration posture here"},
    "elbow_flexed": {"elbow_position": "approximately_90_deg"},
    "elbow_extended": {"elbow_position": "substantially_straighter"},
    "arm_abducted": {"shoulder_position": "upper_arm_away_from_torso"}
  }
}
```

These are illustrative descriptions, not inferred measurements. Block fields are
limited to shoulder/elbow position, forearm orientation and notes. Detailed
positions default to unknown outside reference unless explicitly supplied.
Date defaults to the local recording day, stored as ISO YYYY-MM-DD; recording
start/finish timestamps include the local UTC offset. `--fast --source synthetic`
is available for software checks; its arbitrary gesture patterns are not real EMG.

## Files, trial IDs and validation

Each recording stays in `recordings/<subject>/<session>/`, with the existing
`samples.csv`, `events.csv`, and `metadata.json`. Version 2 appends condition
columns without changing the channel encoding. Every trial has a stable ID:

```text
v2_session_A_reference_fist_r01
v2_session_A_elbow_extended_pinch_r06
```

The existing `trial` column and the new `trial_id` both carry that ID. Sample and
event columns also preserve posture, repetition index, session identity, placement,
forearm orientation, shoulder/elbow position and static-motion status. Metadata
registers full gesture boundaries, duration, trial order, blocks and independent
rest intervals. Reconnect subsegments of one trial share a splitting group.

The recorder validates before marking a session complete. The evaluator validates
again before fitting or writing results. Checks reject missing gestures, incomplete
postures, short/duplicate repetition counts, duplicate IDs, overlapping labels,
inconsistent sample/event boundaries, impossible/short durations, invalid condition
metadata, non-finite channels, incorrect timing and completion disagreement.
Failures leave acquisition metadata incomplete. Short independent rest below
45 seconds and reconnects produce explicit validation warnings. Signal quality
still needs review; structural validation does not certify physiological quality.

Build a separate inventory so the original benchmark manifest is preserved:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
.\.venv\Scripts\python.exe -m emg.evaluation.dataset --recordings recordings --output data/session_manifest_v2.csv
```

The manifest now carries optional `protocol`, `session_role`, `posture_set`,
`repetitions_per_posture`, and `gesture_set_id`, alongside v1.1's optional condition
fields. `gesture_set` still contains exact pipe-delimited labels. For multi-posture
sessions, the manifest's initial `arm_posture` describes reference/baseline;
`posture_set` lists blocks, while sample/trial conditions drive v2 evaluation.
Do not override known recorded placement/date/reapplication/static-motion facts
with manifest annotations. Legacy manifests missing new columns still load.

Validate without computing features or producing performance results:

```powershell
.\.venv\Scripts\python.exe .\dataset_v2_benchmark.py --manifest data/session_manifest_v2.csv --session-a v2_session_A --session-b v2_session_B --validate-only
```

## Evaluation modes

Install the existing optional plotting dependency, then evaluate available sessions:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-benchmark.txt
.\.venv\Scripts\python.exe .\dataset_v2_benchmark.py --manifest data/session_manifest_v2.csv --session-a v2_session_A --session-b v2_session_B --session-c v2_session_C
```

Omit B/C until recorded. In `all` mode, unavailable B/C comparisons are skipped;
requesting an experiment that needs a missing target fails clearly. Use
`--experiment` to select a mode. Every run materializes the reference control for
deltas. Outputs default to `results/dataset_v2/`. Use a fresh `--output` for each
analysis; a nonempty directory is rejected to prevent mixed artifacts.

| Experiment | Training | Held-out evaluation |
|---|---|---|
| `reference-baseline` | Half A reference trials per class (3 of 6), seed-selected | Remaining A reference trials. |
| `cross-posture` | Frozen reference-baseline model | All A trials at each altered posture, separately; no retraining. |
| `posture-inclusive` | Half the trials per class at reference/flexed/extended | Remaining trials at those postures, plus all abducted trials. |
| `reapplication-zero-shot` | Frozen A reference model | All B reference trials, new placement. |
| `recalibration` | See calibration semantics below | Remaining target-session reference trials. |
| `cross-day` | Frozen A reference model | All C reference trials, new day/application. |

Within-condition splits select complete trial IDs per class rather than individual
windows. The same reference partition is used across comparisons. Inclusive
training has `--training-postures` for configuration; it does not change the
classifier. It reports `reference_only_same_test_macro_f1` and
`delta_vs_reference_only_same_test` by scoring the frozen reference model on the
**identical** held-out trials, isolating representative training exposure. Ordinary
cross-posture scores use all altered-posture trials, so their test sets differ from
inclusive held-out subsets; use the paired columns for a fair comparison.

Only `gesture` phases are classification trials, including explicit rest gestures.
Inter-trial rest, normalization-baseline and prepare windows are excluded from
training and classification holdout. Separate long-rest metrics use `rest_pre`
and `rest_post`; `--include-baseline-rest` explicitly adds baseline to that metric.
The scaler and LDA only fit training windows. Train/test trial IDs and indices are
saved and checked disjoint. The canonical continuous-recording filter and short
per-recording baseline normalization are retained, including a target recording's
own short rest calibration at zero-shot. Zero-shot means **zero new labeled
gesture trials**, not absence of baseline normalization.

## Recalibration burden

The default sizes are **0, 1, 2, 3 trials per gesture**, with five deterministic
choices for nonzero sizes. `--calibration-reps 0 1 2 3 4` supports four when enough
repetitions exist. A size that leaves no held-out trial per class is rejected.

- **0:** no B/C gesture trials train; use the frozen A reference model and test all
  target reference trials. This point is evaluated once.
- **1/2/3:** select exactly that many complete trials of **each** class (including
  rest) from B/C. Refit both StandardScaler and uniform-prior LDA. Test all remaining
  target reference trials; no calibration trial appears in evaluation.
- Default **new-only** training discards A for nonzero sizes. Explicit **combined**
  training adds A's original reference training partition to target calibration.
  It does not silently add A's reference holdout or other posture data.

Each choice uses NumPy `SeedSequence([seed, choice])`, then sorted-class trial-ID
permutations. For a given choice, size k uses the first k IDs per class; selections
are nested across sizes. Different classes can use different repetition indices;
these are per-gesture selections, not necessarily a common repetition block.
Duplicate complete selections are skipped and actual distinct-choice counts are
saved. Repetition-choice SD describes selection variability within this dataset,
not independent-subject uncertainty.

All remaining target trials test each size, so test sets shrink and change as k
increases. This directly answers the requested remaining-repetition experiment;
do not interpret the curve as a comparison on one fixed test set. The current
target short normalization baseline remains available at every k. Long independent
rest is never trained and is identical across choices/sizes within a target session.

Reports include actual **active calibration seconds** (excluding explicit rest
trials) and **labeled calibration seconds** (including rest trials). At default
3-second holds, these are 24/48/72 active seconds and 27/54/81 labeled seconds for
1/2/3 repetitions. They omit preparation, inter-trial rest and the five-second
normalization baseline; they are not total user elapsed time.

Explicit combined comparison and optional C recalibration:

```powershell
.\.venv\Scripts\python.exe .\dataset_v2_benchmark.py --manifest data/session_manifest_v2.csv --session-a v2_session_A --session-b v2_session_B --experiment recalibration --calibration-training combined --output results/dataset_v2_combined
.\.venv\Scripts\python.exe .\dataset_v2_benchmark.py --manifest data/session_manifest_v2.csv --session-a v2_session_A --session-c v2_session_C --experiment recalibration --calibration-target C --output results/dataset_v2_cross_day_recalibration
```

## Reports and interpretation

Each result saves `metrics.json`, `summary.csv`, raw/smoothed confusion matrices,
per-class precision/recall/F1/support, `predictions.csv`, `run_config.json`, and an
evaluation model. Predictions retain truth/probabilities, trial/sample provenance,
conditions and `evaluation_scope` (classification or independent rest). F1 and
classification confusion matrices describe held-out gesture trials; long rest is
reported separately. This avoids long-rest class prevalence dominating macro F1.

`results/dataset_v2/summary.csv` lists every result/choice with sessions, postures,
placements, calibration size/duration, class count, F1 and reference delta,
balanced accuracy, switching and rest metrics. `summary_aggregate.csv` supplies
means/sample SDs across recalibration choices. Each experiment retains its own
result subdirectories and exact train/calibration/test IDs. Root/per-result
configuration includes input/source SHA-256 fingerprints, dependency versions,
validation warnings and whether inputs are serial or synthetic.

Independent-rest metrics for raw and smoothed output include:

- False-active window fraction, episode count and episodes/minute.
- Mean/maximum false-activation duration and switches/second.
- Observation exposure, false-active windows and boundary-censored episode counts.
- Separate `rest_pre` and `rest_post` metrics, plus their combined reference-rest score.

An episode is a consecutive run of non-rest predictions; changing active class
does not begin another episode. Duration is run windows x hop duration. Exposure
is scored windows x hop duration, excluding guards, unscored gaps and transitions.
Runs touching observation boundaries are censored and flagged. The two long
recordings provide much more rest exposure than v0's 4.2-second estimate.
High-level `false_active_fraction` is the independent-reference-rest metric;
`trial_rest_false_active_fraction` separately records explicit rest gesture trials
at the actual test posture. Long-rest values do not establish altered-posture rest
robustness, since those long intervals are recorded at reference.

Plots include cross-posture F1, per-gesture posture recall, recalibration mean F1
with choice SD, and same-placement versus zero-shot/recalibrated reapplication.
Each confusion matrix retains all labels; inspect wrist_up/down, radial/ulnar,
fist/pinch, and splayed/point errors directly rather than relying on accuracy.

Configured timing is saved as 200 ms window, 40 ms hop, seven-window smoothing,
240 ms history span and 440 ms complete smoothed sample span. These are configured
spans, **not measured stabilization delay**. Segment-local scoring omits guarded
transitions, so no transition latency measurement is claimed. Wake logic, confidence
gates and browser hand motion are outside these ungated classifier metrics.

No real Dataset v2 performance is reported until compatible recordings exist.
Synthetic runs verify recording, validation, grouping, fitting and artifact
generation only. The next intervention should follow the real reference,
cross-posture and recalibration measurements.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover schedule balance/order, metadata parsing, recording validation,
trial/sample isolation, reconnect grouping, exact calibration counts and nested
choices, all evaluation modes, combined/C recalibration, independent-rest episodes
and durations, and deterministic reports/predictions. Existing tests retain
legacy compatibility and full-precision Baseline v0 equivalence.
# Short recordings with seven gestures

For the current collection, use [COMPACT_RECORDING.md](COMPACT_RECORDING.md).
It reuses an audited completed reference block and records new static positions
in separately saved 21-trial chunks, with a separate 14-trial evaluation chunk.
Radial and ulnar deviation are omitted. The historical full protocol below
retains its original nine-class defaults; the interactive launcher now defaults
to seven classes and enforces a maximum of 99 trials per recording.
