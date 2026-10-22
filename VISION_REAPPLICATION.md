# Seven-gesture electrode reapplication

The prospective protocol is frozen in `data/vision_reapplication_protocol_v1.json`.
The capture mode and evaluator verify its source-file hashes before running. They
do not modify the source recording or saved models.

## Offline checks

Print the 86-step schedule (two baselines and 84 trials) without opening COM10:

```powershell
.\.venv\Scripts\python.exe .\vision_ground_truth.py `
  --reapplication-protocol .\data\vision_reapplication_protocol_v1.json --plan
.\.venv\Scripts\python.exe .\evaluate_vision_reapplication.py --plan
```

The test suite includes a complete synthetic 84-trial fixture that exercises
preprocessing, frozen CNN and LDA scoring, and all three nested adaptation
sizes. It cannot estimate accuracy on new electrodes.

## Record the new application

Use the same participant and nominal electrode placement. Record a new placement
ID and a photo or written landmarks. Check all eight raw channels for flat or
saturated contact before beginning, particularly CH3. Keep the frozen test
results hidden during collection.

Close other serial apps, turn on the transmitter, then run:

```powershell
.\.venv\Scripts\python.exe .\vision_ground_truth.py `
  --source serial --port COM10 `
  --reapplication-protocol .\data\vision_reapplication_protocol_v1.json
```

The local page is `http://127.0.0.1:8767/vision.html`. Calibrate the camera
templates for both orientations. The operator sets the first block to **vertical**;
the software records this operator-supplied posture separately from the camera's
measured orientation. Capture a relaxed five-second baseline, then six rounds
with one accepted trial of each gesture per round. Rejected holds do not count.
After all 42 vertical trials, use **Start horizontal block**, set the forearm
horizontal, capture another baseline, and complete six more rounds.

Stopping early preserves the recording but marks it not analysis-ready. The
evaluator requires two baselines, 84 unique accepted trials, correct posture
order, and intact trial samples. A synthetic session is rejected from scientific
scoring unless the explicit `--allow-synthetic` software-check flag is used.

## Evaluate once the session is complete

Choose a fresh output directory and use the actual session directory:

```powershell
.\.venv\Scripts\python.exe .\evaluate_vision_reapplication.py `
  --session .\recordings\S01\YOUR_NEW_SESSION_ID `
  --output .\results\vision_reapplication_YOUR_NEW_SESSION_ID
```

The evaluator verifies frozen hashes and writes `zero_shot.json` **before any
target fitting**. It scores the untouched CNN and LDA on trials 4–6 in each
posture with (1) the original source baseline scale and (2) only the new
posture baselines. It then fits a fixed, new-only shrinkage LDA classifier on
the frozen CNN's 32-dimensional embeddings using trials 1, then 1–2, then
1–3 per gesture and posture. Every stage uses the same 42 fixed test trials.
`summary.json` and `summary.csv` contain the resulting curve. The primary
strict zero-shot CNN score also includes a whole-trial bootstrap interval,
stratified by gesture and posture. Prediction delay is measured from trial
start to the first correct guarded window. Independent-rest false activation
is reported as unavailable because the frozen schedule has no independent
rest segment; held-out `rest` trials remain in the class metrics.

The first new-application result is a test of transfer. It must not be replaced
by the high training-session replay score, and test feedback must not alter
trial exclusions, preprocessing, thresholds, or the frozen model.
