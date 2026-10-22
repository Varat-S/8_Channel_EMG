# P4 42-trial reapplication study

The original serial recording is `recordings/S01/vision_20261022_202644`. It has 42 accepted trials, one baseline, and is incomplete under the frozen 84-trial protocol. The recorder tagged every trial as vertical. The participant reported that repetitions 1-3 of each gesture were vertical and repetitions 4-6 were horizontal. The analysis copy in `prepared/vision_20261022_202644_3v3h` records that correction without changing the original files. Camera orientation measurements remain in the metadata as an audit signal; some disagree with the reported split.

The old P3 CNN was frozen before this P4 session. Scoring it on all 42 P4 trials gives smoothed accuracy 0.749 and macro F1 0.728 with the original P3 baseline. Replacing only the baseline with the fresh P4 baseline gives accuracy 0.569 and macro F1 0.503. These are exploratory transfer results because this recording did not follow the frozen 84-trial protocol.

The new P4 CNN was evaluated with whole-trial grouped folds on this session: mixed-posture three-fold smoothed accuracy 0.822 and macro F1 0.822. The final model in `models/vision_posture_20261022_202644_3v3h/cnn_all_postures.pt` was then fit on all 42 trials. Those evaluation numbers are within-session estimates, not accuracy of the final model on future live gestures.

The hand-only local viewer runs with:

```powershell
.\.venv\Scripts\python.exe .\vision_live.py --port COM10 --model .\models\vision_posture_20261022_202644_3v3h\cnn_all_postures.pt --baseline-session .\prepared\vision_20261022_202644_3v3h --baseline-mode original
```

Open `http://127.0.0.1:8768/hand_only.html`. It reuses the recorded P4 baseline and saves no new EMG samples or predictions. A continuous receiver stream is still required; a single readable frame does not establish that the transmitter remains streaming.
