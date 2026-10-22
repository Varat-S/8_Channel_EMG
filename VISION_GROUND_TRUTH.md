# Vision-assisted EMG ground truth

This tool records synchronized EMG while a local webcam pipeline proposes the
gesture label. It is deliberately separate from the live LDA application so a
camera or browser failure cannot overwrite a trained model.

## What it records

Each session contains:

```text
recordings/<subject>/<session>/
  samples.csv                 raw EMG with sample-indexed phase and label
  events.csv                  accepted boundaries and reconnect events
  metadata.json               protocol, counts, thresholds, accepted/rejected trials
  vision_observations.jsonl   landmarks, confidence, timing and geometry metrics
```

No camera images or video are saved. MediaPipe runs in the local browser from
vendored model/runtime files. The Python server receives numeric landmarks and the
proposed label only.

The acquisition vocabulary is:

- `rest`: neutral, relaxed-looking hand and straight wrist
- `fist`: the four fingers flexed
- `splayed`: the four fingers extended and visibly separated
- `peace`: index and middle fingers extended, ring and little fingers closed
- `point`: index finger extended, other fingers closed
- `pinch`: thumb and index tips touching
- `thumbs_up`: four fingers closed with the thumb extended

`rest` is only a visual posture label. A camera cannot prove physiological muscle
relaxation, so baseline signal quality still needs EMG review.

## Capture state machine

1. The user enables the camera and runs one prompted visual calibration. Eighteen
   stable landmark examples are collected for every pose first with a vertical
   forearm and then with a horizontal forearm (14 prototypes total).
2. The page builds personal pose templates from four finger-curl values, three
   normalized fingertip gaps, thumb curl, thumb-to-index distance, and normalized
   thumb reach. Hand distances use MediaPipe's 3D world landmarks rather than
   perspective-sensitive image distances.
3. The user explicitly starts an EMG session. Collection after this point is
   recognition-triggered and poses may be presented in any order.
4. A stable visual `rest` hold starts a five-second baseline.
5. After baseline, any label stable for at least 1.1 seconds starts a 2.5-second
   candidate EMG segment.
6. The server independently checks every EMG sample against the freshest camera
   observation.
7. A candidate is accepted only if label/confidence remain valid and the
   elbow-wrist chain stays visible and stable, with at most 0.30 seconds of invalid
   observations. When the shoulder is visible, elbow flexion is also measured;
   otherwise stability uses elbow location and forearm orientation. Rejected data
   is marked `prepare` and the existing dataset loader excludes it.
8. Return to neutral before repeating. Gestures can be presented in any order;
   automatic capture for a label stops when its displayed target is reached.

Classification uses distance to seven personal pose classes, with two posture
prototypes per class, rather than global gesture thresholds. A pose becomes `unknown` when it is too far from every template
or the two nearest templates are ambiguous. This is particularly important for
`rest` versus `splayed`, `peace` versus `point`, and `fist` versus `thumbs_up`.
Each gesture has vertical and horizontal prototypes; the nearer one is used, while
diagonal forearm positions interpolate naturally through feature space. Recalibrate
whenever camera orientation or framing changes substantially.

Each accepted trial stores mean elbow angle, forearm elevation, wrist deflection,
elbow posture bucket, and forearm orientation in metadata. The posture buckets are
also written into the sample CSV as conditions; they are not treated as gesture
classes.

The browser display target is a collection aid, not an automatic stopping rule.
Use **Stop & save** after reaching the intended repetitions.

## Running it

Install/copy browser assets once:

```powershell
npm.cmd install
```

Camera-only dry run:

```powershell
.\.venv\Scripts\python.exe .\vision_ground_truth.py --source synthetic
```

Live COM10 acquisition:

```powershell
.\.venv\Scripts\python.exe .\vision_ground_truth.py --source serial --port COM10 --baud 921600
```

Turn the transmitter on before clicking **Start EMG session**. If it has been off
between runs, wait for the receiver connection to recover before presenting the
baseline. Keep the hand, wrist, and elbow visible; the shoulder is optional but
enables true elbow-flexion measurement. Use a consistent side/oblique camera angle
throughout the visual calibration and recording.

Useful options:

```text
--http-port 8767
--recordings-root recordings
--baseline-seconds 5
--trial-seconds 2.5
--vision-confidence 0.62
--no-browser
```

## Validation policy

Treat the first session as a pilot. Before training:

- inspect the saved accepted/rejected counts and observation metrics;
- manually review several transitions while watching the on-page label;
- verify that `fist` and `thumbs_up` remain distinct at the chosen camera angle;
- run the existing session-quality and grouped evaluation tools;
- keep at least one manually checked session out of training.

The system removes transcription and timing work, but it does not make visual
heuristics infallible. The audit trail is retained specifically so incorrect
labels can be detected rather than silently entering the model.

## Tests

```powershell
npm.cmd run test:vision
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_vision_capture.py -v
Remove-Item Env:PYTHONPATH
```
