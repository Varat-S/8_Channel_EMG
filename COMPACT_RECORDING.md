The compact protocol uses seven classes: rest, fist, splayed fingers, wrist
extension (`wrist_up`), wrist flexion (`wrist_down`), thumb–index pinch, and
index pointing. Radial and ulnar deviation are excluded from new collection
and from the completed reference-trial index. Historical raw files stay intact.

The current application is `P2_20261006_140659`, on the left arm. The user
confirmed the electrodes stayed on and unchanged after the interrupted
`v2_A_20261006_140659` recording. Its reference is the forearm perpendicular
to the ground. There are 42 completed trials of the seven selected classes,
six per class. Three have complete metadata/sample boundaries but missing
event-file boundary records following interruption; the collection index
records that evidence explicitly. There is a 60-second rest_pre, but no
rest_post. The original session remains incomplete.

The extended-arm chunk has now been recorded. The participant confirmed keeping
the arm extended throughout, including the baseline and both independent rests.
Those reference labels are therefore physically incorrect. The recording folder
contains `condition_corrections.json` with confirmed posture overrides and original
file hashes. An analysis adapter must apply those corrections explicitly; the
default Dataset v2 validator refuses to evaluate this chunk using its original
reference labels. Its baseline is posture-specific, which must be reported when
comparing it with recordings normalized in the vertical reference position.
The third labelled rest trial was interrupted by a reconnect and is flagged for
exclusion from clean comparisons. The six active classes each retain all three
uninterrupted holds. See `results/recording_audits/compact_A_20261006_1_extended/`
for the audit and signal plot, including the strong 50 Hz component in relaxed data.

| Chunk | Static condition | Repetitions per class | Trials |
| --- | --- | --- | --- |
| Existing reference | Vertical forearm, neutral orientation | 6 | 42 already recorded |
| 1: extended | Arm down beside body, elbow comfortably nearly straight, neutral | 3 | 21 |
| 2: horizontal | Bent elbow beside body, horizontal forearm, palm down | 3 | 21 |
| 3: side | Arm comfortably away from torso, palm up | 3 | 21 |
| 4: unfamiliar | Different diagonal position and height, neutral | 2 | 14, evaluation only |

Initial collection is **77 new trials**, with 63 for training and 14 reserved
for checking generalization. Each 21-trial chunk takes about six minutes at
the nominal sampling rate, including baseline and rest. The 14-trial check
takes about five minutes. Hardware pauses can extend these times. No new
chunk repeats gesture trials in the original vertical reference position.

Start the menu from the EMG directory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\record_compact_chunks.ps1 -CollectionPath .\data\collection\compact_A_20261006\collection.json
```

Choose 1, 2, or 3. Each launcher waits for readiness, checks the receiver,
then waits for Enter again before starting. Begin in the original reference
position, relax for a five-second normalization baseline and 60-second rest,
then follow the position instruction. Each gesture has three seconds of
preparation, a three-second hold, and two seconds of relaxation. Return to
reference for the final 60-second rest. Keep electrodes unchanged between
chunks and take breaks. Every completed trial and rest block flushes its
sample/event files before its progress is saved. The launcher rejects any
plan above 99 trials. Interrupted chunks remain incomplete.

Wrist extension and flexion are relative to the forearm, so the gesture stays
the same when the palm faces down or up. Use a comfortable, consistent effort.
Keep fingers relaxed during the wrist gestures; keep thumb position consistent
during pointing and other fingers consistent during pinch. If a hold is wrong,
record its displayed trial number for exclusion; do not relabel it as correct.

Three repetitions per new position are a starting budget, not an established
minimum for this hardware/person. Position diversity has evidence behind it:
[Beaulieu et al., multi-position myoelectric training](https://pubmed.ncbi.nlm.nih.gov/28983183/).
That evidence does not establish a universal 100-trial sufficiency threshold.
The classification unit for splitting is a whole gesture trial. Overlapping
windows are not independent repetitions.

After collection, use the original decoder's features, scaler/LDA and smoothing
to compare reference-only and posture-inclusive training on identical held-out
trials. Evaluate each gesture and each condition, plus independent-rest false
activations. Keep chunk 4 out of training and tuning. Its 14 trials provide an
initial screen, not a precise estimate of performance at all possible arm
positions. If performance is weak, collect another 21-trial chunk at the weak
condition before collecting a fresh evaluation chunk. Once an evaluation
result guides tuning or extra collection, that evaluation is no longer an
untouched final test. Avoid collecting every possible extra chunk upfront.

`collection.json` stores source hashes, selected original trial boundaries,
missing event evidence, and separate training/evaluation purposes. Preparation
does not train a model or bypass incomplete-session validation. The existing
full Dataset v2 benchmark requires complete sessions and does not pool these
chunks automatically. Analysis must explicitly adapt the audited reference
block and pool the training chunks with trial/condition provenance; preserve
the missing rest_post and interrupted-source status in its report. Each new
completed chunk itself passes the Dataset v2 recording and feature-loader
checks. The decoder's canonical processing has not changed.

For another application, do not reuse these prepared conditions or placement
ID. Prepare and record a new reference under its actual application identity.
The baseline and independent rest in a posture chunk remain in reference;
they do not measure rest false activations at every altered posture.
