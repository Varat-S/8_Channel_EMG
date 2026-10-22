The seven-class model is saved as `models/compact_A_20261006_v1.joblib`.
It uses 103 retained whole trials from the interrupted reference block and the
three complete training chunks. Radial/ulnar deviation and two interrupted holds
were excluded. Sample/event/source hashes and selected trial IDs are recorded in
the artifact and `results/compact_model_v1/run_config.json`. Chunk 4 was never used
for training, parameter selection or refitting.

The adapter verifies the audited reference source, keeps its interrupted status
in provenance, and uses only registered complete selected holds. It applies the
extended-arm condition corrections in memory. It resets causal filters at recorded
reconnect points and guards the edges of each uninterrupted segment. Features are
the original 32 MAV/WL/ZC/SSC features, with a StandardScaler and uniform-prior LDA.
Splits keep entire gesture trials together, including every overlapping window.
Independent long rests are scored separately and never used as gesture training
windows.

Four processing candidates were compared on five reproducible whole-trial
validation draws:

| Candidate | Mean smoothed macro F1 |
| --- | ---: |
| Per-recording baseline normalization | 0.713 |
| Shared original-reference normalization | 0.800 |
| 50 Hz notch, per-recording baseline normalization | 0.661 |
| 50 Hz notch, shared original-reference normalization | 0.797 |

The selected model uses the shared original five-second reference RMS scale for
all training chunks and live input. This removes the dependence on the differing
baseline conditions of each chunk. It is intended for the unchanged electrode
application; it does not estimate a new live normalization scale when connecting.
The bandpass remains causal fourth-order 10–55 Hz at nominal 125 Hz, with CAR,
25-sample windows, five-sample hop and seven-window probability smoothing.
The tested notch is a causal 50 Hz, Q=30 filter using
[SciPy iirnotch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html).
It was not selected because it did not improve mean validation performance.

The illustrative seed-7 internal split scored 0.820 smoothed macro F1 and 82.4%
balanced accuracy. Reference-only training scored 0.617 macro F1 on exactly the
same held-out trials. These are internal model-selection results, not independent
test accuracy. The deployable model was then fit on all 103 retained training
trials; its training performance is not reported as accuracy.

After the model was frozen, chunk 4 was evaluated with no retraining or retuning:

- 74.8% balanced accuracy and 0.741 smoothed macro F1.
- Correct majority prediction on 11 of 14 gesture trials.
- 0.82% false-active windows over the separate long relaxed recordings.

See `results/compact_model_v1_unfamiliar/` for the confusion matrices, per-class
metrics, all predictions, trial-majority scores and frozen model hash. The test is
small and covers one unfamiliar static condition under the same electrode
application. It does not establish reliable recognition at arbitrary positions,
during movement, after reapplying electrodes, or on another day. Since the test has
now been inspected, subsequent tuning needs a new untouched final test.

Launch with the matching compact live runner from the EMG folder:

```powershell
.\.venv\Scripts\python.exe .\compact_webapp.py --model .\models\compact_A_20261006_v1.joblib --http-port 8766
```

Open `http://127.0.0.1:8766` and click **Test saved model** if recognition is not
already running. Unlock with **splayed → fist → splayed**, holding each pose
steadily for at least 0.4 seconds. The display supports rest, fist, splayed,
wrist extension, wrist flexion, pinch and point. The preview buttons only preview
poses; they are not EMG commands. Use **Reset connection** to release COM10 before
recording further chunks. Fresh single-position calibration is disabled on this
page so it cannot replace the multi-posture model.

Use this runner for this artifact: its compact processing profile stores the
fixed scale, any selected notch, reconnect guard and smoothing settings. Other
live tools do not automatically understand that profile. Existing model files
and historical benchmark outputs were retained.

To reproduce training under fresh output/model paths:

```powershell
.\.venv\Scripts\python.exe .\train_compact_model.py --collection .\data\collection\compact_A_20261006\collection.json --output .\results\compact_model_repeat --model-output .\models\compact_A_repeat.joblib
```
