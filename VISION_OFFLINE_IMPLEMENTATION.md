# Vision offline preparation follow-up

This implements the four remaining items named in
`results/vision_posture_20261016_192456/OFFLINE_PREPARATION.md`. The original
recording, posture correction, frozen model, and held-out report were not changed.

## Local live inference

Run the local site with the frozen seven-gesture CNN and its original P3 baseline
scale, without a new baseline or saved recording:

```powershell
.\.venv\Scripts\python.exe .\vision_live.py --port COM10
```

Open `http://127.0.0.1:8768/vision_live.html`. The site shows the prediction,
confidence, all seven probabilities, frozen P3 baseline RMS, and live channel RMS
relative to that baseline. Check electrode contact and amplitudes, especially
CH3, before relying on the display. A prediction below the 0.55 confidence floor
is `uncertain`. The server uses 10–55 Hz causal filtering, common-average
reference, 25-sample windows, five-sample hops, and seven-window causal
probability smoothing. It opens the serial port only after **Start inference**.
`--baseline-mode fresh` is available if a fresh five-second rest baseline is
desired; that is baseline adaptation, not strict zero-shot inference.
Stop the recording app before starting live inference, because both need the same
receiver.

For a hardware-free site and stream check:

```powershell
.\.venv\Scripts\python.exe .\vision_live.py --source synthetic
```

Synthetic predictions check the software path only.

## Held-out benchmark and trial uncertainty

Reproduce the grouped three-fold benchmark with:

```powershell
.\.venv\Scripts\python.exe .\benchmark_vision_offline.py --output .\results\vision_offline_benchmark_new
```

The results in `results/vision_offline_benchmark_v3/` use the same whole-trial
mixed-posture folds and fixed epoch counts as the original evaluation. The CNN
is trained on each fold's clean training trials; corruption is applied only to
its held-out windows **after preprocessing**. Each preprocessing ablation is
retrained and retested on the same folds, for both the CNN and LDA reference.

| Evaluation | Smoothed macro F1 |
|---|---:|
| Clean held-out CNN | 0.833 |
| Half global gain | 0.187 |
| Drop CH3 | 0.433 |
| No common-average reference, retrained CNN | 0.789 |
| No band-pass, retrained CNN | 0.132 |
| No baseline normalization, retrained CNN | 0.832 |

The clean score's 95% percentile bootstrap interval is **0.766–0.890**. Each of
4,000 bootstrap draws resamples complete trials within gesture and posture; it
does not resample overlapping windows independently. The interval describes
this recording and split, not uncertainty for a future electrode application.
The near-equality of the baseline-normalization ablation is expected because
each model is refit with training-fold normalization, which can absorb a fixed
session scale. It does not establish that the frozen deployment model can omit
a fresh baseline. Artificial post-preprocessing corruptions do not model all
effects of a physical electrode change.

## Representation experiments

Reproduce the fixed-weight exploratory experiments with:

```powershell
.\.venv\Scripts\python.exe .\experiment_vision_representation.py --output .\results\vision_representation_new
```

`results/vision_representation_v2/` compares two auxiliary losses on the same
whole-trial folds and fixed epoch counts: supervised contrastive positives
share gesture but differ in posture, and a regression head predicts channel
RMS in a nonoverlapping window 200 ms ahead. The auxiliary weight is fixed at
0.1; neither experiment uses held-out trials to tune it.

| Model | Smoothed macro F1 |
|---|---:|
| Original supervised CNN report | 0.832 |
| Cross-posture contrastive auxiliary loss | 0.834 |
| Future RMS auxiliary loss | 0.832 |

The differences are too small to establish a benefit on this single session.
These experiments are not deployment candidates or cross-application accuracy
claims. The prospective seven-gesture reapplication protocol remains the next
test of transfer.
