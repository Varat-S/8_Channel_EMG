# EMG project status and change log

Updated: 2026-10-22. This is the working inventory for the current seven-gesture vision-assisted EMG study. Update this file when a dataset, model, evaluation, or live feature changes. For implementation details, see `README.md`, `VISION_GROUND_TRUTH.md`, and `REAPPLICATION_PROTOCOL.md`.

## Current study and data

The seven labels are `rest`, `fist`, `splayed`, `peace`, `point`, `pinch`, and `thumbs_up`. The participant is S01. EMG is received on COM10 at 921600 baud; the decoded stream is nominally 125 complete frames/s from eight channels. The ADC scale remains provisional, so values are treated as counts. The browser camera estimates hand and forearm landmarks locally; the operator's posture block remains the protocol truth.

| Application | Recording | Trials and posture information | Role |
| --- | --- | --- | --- |
| P3 | `recordings/S01/vision_20261016_192456` | 42 trials; for each gesture, repetitions 1–3 vertical and 4–6 horizontal | First training session and frozen original reapplication model |
| P4 | `recordings/S01/vision_20261022_202644` | 42 trials, one baseline; recorder tagged all trials vertical | Second application; original recording retained |
| P4 analysis copy | `prepared/vision_20261022_202644_3v3h` | User-reported repetitions 1–3 vertical, 4–6 horizontal; camera estimates remain available for audit | Analysis and training input |
| P5 | Not yet collected | Frozen protocol requires a new application, two posture baselines and 84 accepted trials | Prospective test |

P4 did **not** satisfy the earlier frozen 84-trial protocol. Its transfer results are exploratory. The corrected P4 posture assignment follows the participant's report; it was not independently confirmed by the recorder's block tags, and some camera estimates disagree.

## Models and measured results

All metrics in this table are *smoothed window* metrics with whole trials held out where folds are specified. Accuracy of a model fitted on all available trials has not been measured on a future electrode application.

| Model or test | Evaluation | Smoothed accuracy | Smoothed macro F1 | Artifact or report |
| --- | --- | ---: | ---: | --- |
| P3 CNN | P3 grouped folds | 0.836 | 0.832 | `models/vision_posture_20261016_192456/cnn_all_postures.pt`; `results/vision_posture_20261016_192456/REPORT.md` |
| Frozen P3 CNN → P4 | All 42 P4 trials, original P3 baseline | 0.749 | 0.728 | `results/vision_posture_20261022_202644_3v3h/REPORT.md` |
| Frozen P3 CNN → P4 | All 42 P4 trials, fresh P4 baseline | 0.569 | 0.503 | Same P4 report |
| P4 CNN | P4 grouped folds | 0.822 | 0.822 | `models/vision_posture_20261022_202644_3v3h/cnn_all_postures.pt` |
| P3+P4 plain CNN | Paired whole-trial folds across both applications | 0.840 | 0.840 | `models/vision_multisession_P3_P4_v1/cnn_plain.pt` |
| P3+P4 gain augmented CNN | Same paired folds | 0.823 | 0.821 | `models/vision_multisession_P3_P4_v1/cnn_gain.pt` |
| P3+P4 gain and channel-dropout CNN | Same paired folds | 0.822 | 0.819 | `models/vision_multisession_P3_P4_v1/cnn_gain_dropout.pt` |

The pooled comparison is in `results/vision_multisession_P3_P4_v1/summary.json` and `.csv`. Folds hold out entire trials from both applications, pairing repetitions (1,4), (2,5), and (3,6). This estimates performance with the same two applications represented in training; it is not a test on a third application. Per-session pooled plain CNN F1: P3 0.828, P4 0.851. The plain model is the **preselected primary P5 candidate**. Gain and gain plus dropout are secondary candidates. Gain augmentation multiplies each channel by a sampled factor from 1/1.5 to 1.5 and the entire window by 1/1.25 to 1.25 during training. The dropout variant also zeros one channel in 15% of training trials. Augmentation did not improve these folds.

The short calibration pilot is in `results/vision_fewshot_P3_P4_v1/summary.json` and `.csv`. It uses a frozen source CNN as a feature extractor, then fits a new shrinkage LDA from labeled trials in the target application. With source baseline scaling, P3 → P4 on fixed repetitions 3 and 6 (14 trials) gave F1 0.671 at zero calibration, 0.711 after 14 calibration trials (one per gesture and posture), and 0.795 after 28 (two per gesture and posture). Reverse P4 → P3 gave F1 0.232, 0.605, and 0.662. This is a two-application pilot with a split chosen after both datasets existed. The gains are promising but not a prospective accuracy estimate.

## Available features

| Feature | Entry point | What it does |
| --- | --- | --- |
| Camera-only old site | `vision_ground_truth.py --view-only --no-browser` | Runs the vision-assisted page at `http://127.0.0.1:8767/vision.html`, with no EMG acquisition. Only one page can own the camera at a time. |
| Live hand-only demo | `vision_live.py` | Runs the articulated seven-gesture hand at `http://127.0.0.1:8768/hand_only.html`. Connect opens COM10; the server alone does not collect training data. |
| Protocol recorder | `vision_ground_truth.py --source serial --port COM10 --reapplication-protocol data/vision_reapplication_protocol_v1.json` | Guides the complete 84-trial P5 capture and validates posture blocks. |
| Original P3 reapplication evaluator | `evaluate_vision_reapplication.py` | Scores frozen P3 CNN and LDA on a completed new application, writes zero-shot results before adaptation. |
| Pooled P3/P4 experiments | `experiment_multisession_vision.py` | Trains and compares plain, gain, and gain plus channel-dropout CNNs using grouped folds. |
| Two-way short calibration pilot | `experiment_fewshot_vision.py` | Compares P3 → P4 and P4 → P3 frozen-feature calibration. |
| Prospective pooled candidate evaluator | `evaluate_p5_candidates.py` | Checks the model hashes, scores all frozen candidates without gesture-label adaptation, saves `zero_shot.json`, then evaluates 14, 28, and 42 labeled calibration trials on the same 42 fixed test trials. |

For the selected pooled CNN in the live hand-only demo, use:

```powershell
.\.venv\Scripts\python.exe .\vision_live.py --port COM10 --http-port 8768 --model .\models\vision_multisession_P3_P4_v1\cnn_plain.pt --baseline-session .\prepared\vision_20261022_202644_3v3h --baseline-mode original
```

This demo reuses P4's recorded rest baseline and requires a continuous receiver stream. Its visual output is not an independent accuracy measurement. To restore the P4-only model, use the launch command in `results/vision_posture_20261022_202644_3v3h/REPORT.md`.

For a public checkout without the private recordings or prepared P4 baseline, start the same model with `--baseline-mode fresh` and omit `--baseline-session`. The viewer will acquire a new five-second relaxed rest baseline when inference starts. Its accuracy is unmeasured; the fresh-baseline transfer result for the older P3 model does not predict this pooled model's live accuracy.

As of the 2026-10-22 deployment check, the pooled plain CNN hand-only server is serving port 8768 and is idle until **Connect** is pressed. The camera-only page is serving port 8767. Server processes are transient; check both URLs after a restart.

## Frozen P5 plan

`data/vision_reapplication_protocol_v1.json` defines 7 gestures × 6 repetitions × 2 postures = **84 accepted gesture trials**, plus one relaxed rest baseline before each posture block. The first three repetitions of every gesture/posture are reserved for calibration (42 trials); repetitions 4–6 are the fixed test (42 trials). A rest/fist contact pilot and camera-template calibration may also be needed before capture but do not count toward the 84 trials. Rejected holds can add attempts.

The pooled models, selected primary candidate, and hashes are frozen in `models/vision_multisession_P3_P4_v1/manifest.json`. The original P3 protocol also hashes the original data and models. Collect P5 with the actual vertical block followed by the horizontal block; use the recorder's **Start horizontal block** transition and collect both baselines. Do not repurpose the 42-trial P4 recording as P5.

Before recording, inspect the planned steps:

```powershell
.\.venv\Scripts\python.exe .\vision_ground_truth.py --reapplication-protocol .\data\vision_reapplication_protocol_v1.json --plan
```

After a **complete new P5 application**, evaluate into fresh result directories:

```powershell
.\.venv\Scripts\python.exe .\evaluate_vision_reapplication.py --session .\recordings\S01\YOUR_P5_SESSION --output .\results\vision_reapplication_YOUR_P5_SESSION
.\.venv\Scripts\python.exe .\evaluate_p5_candidates.py --session .\recordings\S01\YOUR_P5_SESSION --output .\results\vision_p5_candidates_YOUR_P5_SESSION
```

The primary pooled-model outcome is zero-shot, trial-grouped smoothed macro F1 on P5's 42 fixed test trials with only the fresh rest baselines used for normalization. The two augmented variants and the calibration curve are secondary. The original P3 evaluator supplies a separately frozen historical reference. P5 results should be interpreted before changing models or thresholds.

## Change log

- **2026-10-22:** Recorded P4 reapplication, preserved its raw files, and made a corrected 3-vertical/3-horizontal analysis copy based on the participant report. Evaluated frozen P3 transfer and trained a P4-specific CNN.
- **2026-10-22:** Added two-application pooled CNN training with gain and channel-dropout experiments, whole-trial folds, and model artifacts. Selected the plain CNN based on the development folds and froze model hashes before P5.
- **2026-10-22:** Added two-way labeled calibration pilot and the P5 pooled-candidate evaluator. Enabled the hand-only live server to use a recorded baseline from either pooled training session.
- **2026-10-22:** Added this inventory. Record future model, data, protocol, and live-site changes here with date, result, artifact path, and known limitation.
- **2026-10-22:** Deployed the selected pooled plain CNN to the hand-only page on port 8768. Confirmed the page and configuration respond; did not perform a new live hardware accuracy trial.
- **2026-10-22:** Updated the hand-only page to display its actual model artifact and recorded baseline session. Verified the model, baseline, page, and script through the local server; the project suite passed 117 tests.
- **2026-10-22:** Prepared the public GitHub repository with code, documentation, selected model artifacts, and result summaries. Local raw recordings, prepared participant data, prediction traces, the proprietary lab executable, and runtime files are excluded from Git.
- **2026-10-22:** Published the reviewed project to `https://github.com/Varat-S/8_Channel_EMG` on `main`. Future study updates should update this log and be committed with the related code or results.
