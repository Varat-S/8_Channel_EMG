# Five-class shrinkage-LDA candidate

The live-test candidate is
`models/compact_A_20261006_five_core_shrinkage_v2.joblib`. It preserves the original
seven-class model and uses only `rest`, `fist`, `splayed`, `wrist_up`, and
`wrist_down`.

The classifier is StandardScaler plus shrinkage LDA with automatic covariance
shrinkage and uniform class priors. It was fitted on 73 complete trials (4,015
windows) from the reference and three multi-posture training recordings. The signal
pipeline remains fixed-reference normalization, common-average reference, causal
10-55 Hz filtering, 25-sample windows, five-sample hops, and seven-window smoothing.

Output confidence rejection was selected using leave-one-recorded-position-out
predictions only. Predictions below 82% confidence hold the last accepted output;
each new connection/reconnection or lock begins at rest. The selection constrained
active-window coverage to at least 70%, avoiding the more accurate-looking but
impractically sticky 90% threshold.

On the previously inspected unfamiliar-position recording, used only as a
development check, rejection changed smoothed macro F1 from 0.885 to 0.892, balanced
accuracy from 0.889 to 0.898, and independent-rest false-active windows from 0.58%
to 0%. Nine of ten five-class trials had the correct majority output. These are not
untouched final-test results because the session had already been inspected during
earlier model development.

Run the separate candidate interface with:

```powershell
.\.venv\Scripts\python.exe .\compact_webapp.py `
  --model .\models\compact_A_20261006_five_core_shrinkage_v2.joblib `
  --http-port 8766
```

The interface remains locked until `splayed -> fist -> splayed`. The detected label
continues to show the classifier's raw choice. When confidence is below 82%, the
detected label is marked `HELD` and the animated output stays at its last accepted
pose.

Full selection and development-check outputs are in
`results/five_core_candidate_v2/`. A new recording not used anywhere in model or
threshold development is still required for a defensible final evaluation.
