# Vision-assisted posture evaluation

Session: `vision_20261016_192456`

## Dataset

- 7 classes: fist, peace, pinch, point, rest, splayed, thumbs up
- 6 accepted repetitions per class (42 total)
- Repetitions 1–3: vertical forearm
- Repetitions 4–6: horizontal forearm
- 1,806 usable 200 ms windows after 300 ms guards at both trial edges
- Whole repetitions are kept intact across every train/test boundary

## Preprocessing

- 10–55 Hz causal Butterworth band-pass
- Common-average spatial reference across eight channels
- Per-channel RMS normalization from the session baseline
- 25-sample (200 ms) windows, 5-sample (40 ms) hop
- Seven-window causal probability averaging, reset at trial boundaries

## Results

| Model | Evaluation | Raw accuracy | Raw macro-F1 | Smoothed accuracy | Smoothed macro-F1 |
|---|---|---:|---:|---:|---:|
| LDA | Vertical → horizontal | 0.442 | 0.399 | 0.439 | 0.390 |
| LDA | Horizontal → vertical | 0.309 | 0.286 | 0.287 | 0.263 |
| LDA | Mixed-posture grouped 3-fold | 0.678 | 0.675 | 0.744 | 0.738 |
| Compact CNN | Vertical → horizontal | 0.540 | 0.535 | 0.615 | 0.612 |
| Compact CNN | Horizontal → vertical | 0.414 | 0.398 | 0.440 | 0.425 |
| Compact CNN | Mixed-posture grouped 3-fold | 0.770 | 0.767 | 0.836 | 0.832 |

## Interpretation

The large loss when training on one forearm orientation and testing on the other
confirms a substantial posture-domain shift. Including both orientations in
training is materially better. The compact CNN is the stronger candidate in the
mixed-posture experiment, improving smoothed macro-F1 from 0.738 (LDA) to 0.832.

The CNN's weakest mixed-posture classes are splayed (recall 0.636) and fist
(recall 0.655). Its largest confusions are splayed → pinch (72 windows), fist →
point (55), pinch → splayed (48), and fist → thumbs up (25).

## Scope and limitations

These are within-session, whole-repetition held-out results with the same electrode
application. They are not accuracy estimates for a future electrode reapplication,
another day, or another participant. The all-data artifacts are deployment
candidates and their training-set performance is not reported as test accuracy.

Contrastive posture alignment and next-100-ms predictive pretraining were not used
in this run; they remain potential extensions if cross-session/reapplication data
show that the current supervised models do not transfer adequately.
