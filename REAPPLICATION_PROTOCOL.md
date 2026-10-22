# Five-core electrode-reapplication protocol

This protocol measures what survives a new electrode application. It is designed
for the next time the electrodes are attached; none of the preparation or software
checks below opens COM10.

## What is already reusable

The acquisition decoder, filtering, feature extraction, web interface, wake
sequence, and evaluation code carry over unchanged. The uncertain parts are the
old baseline scale and the LDA decision boundaries, because a small change in
electrode position can change which muscles dominate each channel.

The offline synthetic audit in `results/five_core_channel_shift_v1/` supports this
caution but does not estimate real reapplication accuracy. On the previously
inspected development recording, the frozen live policy scored 0.892 macro F1 when
unaltered. Cyclically rotating channel identities reduced it to 0.098--0.380.
Zeroing one channel ranged from 0.627 to 0.905, with CH5 most harmful, and halving
all raw amplitudes reduced it to 0.394. These are deliberately artificial
perturbations, not physiological measurements.

## Before attaching electrodes

Keep the old model and its results unchanged. Prepare a placement record containing:

- a clear forearm photo showing all eight electrodes and the reference electrode;
- the CH1 marker and the direction in which CH1--CH8 proceed around the forearm;
- distance from a fixed landmark (wrist crease or elbow crease) to the electrode
  centre line, forearm circumference there, and centre-to-centre spacing;
- reference-electrode location, transmitter orientation, and cable direction;
- arm, wrist, and palm orientation used as the `reference` posture;
- skin preparation and any electrode that had weak adhesion.

Channel identity is especially important: do not compensate for a shifted physical
placement by plugging the channels into a different numerical order.

The recording schedule can be checked at any time without hardware:

```powershell
.\.venv\Scripts\python.exe .\calibrate.py `
  --protocol dataset_v2 --source serial --session-role B `
  --conditions-json .\data\reapplication_five_core_P3.example.json --plan
```

It should show 30 trials: six each of `rest`, `fist`, `splayed`, `wrist_up`, and
`wrist_down` in randomized within-repetition order.

## Recording P3

Do not run this command until the electrodes are attached, the transmitter is on,
and COM10 is connected. Before recording, copy the example conditions file to a
new file if the session or placement ID needs to change; never reuse an existing
session directory.

```powershell
.\.venv\Scripts\python.exe .\calibrate.py `
  --protocol dataset_v2 --source serial --port COM10 --session-role B `
  --conditions-json .\data\reapplication_five_core_P3.example.json
```

Use the same reference posture throughout. The programmed duration is about 6
minutes 20 seconds: 15 seconds initial settling, 5 seconds baseline, two 60-second
independent rests, and 30 trials of 3 seconds prepare + 3 seconds hold + 2 seconds
rest. Human response time adds a little overhead. Do not test the model or inspect
class performance midway through this session.

## Frozen analysis

The evaluator's design is fixed now, before P3 exists:

- For each class, a seeded permutation assigns three trials to a calibration pool
  and three to a fixed evaluation set.
- Strict zero-shot uses the untouched candidate and its original fixed scale.
- Baseline-only uses the untouched candidate with the new five-second baseline.
- One, two, and three nested calibration trials per class refit shrinkage LDA in
  both `new_only` and `combined` modes.
- Every row is scored on the same 15 evaluation trials. No P3 result changes the
  split, model family, confidence threshold, or selected metric.
- The primary metric is smoothed macro F1. Independent-rest false activation and
  per-class metrics are retained. The existing 0.82 hold rule is supplemental and
  is not retuned on P3.

Review the frozen plan offline with:

```powershell
.\.venv\Scripts\python.exe .\evaluate_five_core_reapplication.py --plan
```

After a successful recording, evaluate it with:

```powershell
.\.venv\Scripts\python.exe .\evaluate_five_core_reapplication.py `
  --session .\recordings\S01\five_core_reapply_P3 `
  --output .\results\five_core_reapplication_P3_v1
```

The command never replaces the saved candidate model. It writes `summary.csv`,
`details.json`, and `run_config.json` to a fresh result directory. If the actual
session ID differs, update `--session` and use a correspondingly named fresh output
directory.

P3 is an adaptation study, not the last unbiased performance claim: once its three
calibration trials per class influence a chosen deployment strategy, a later P4
application should be reserved as the untouched confirmation session.
