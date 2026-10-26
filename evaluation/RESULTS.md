# Sleep Detection — Preliminary Evaluation

## Method

The prototype uses MediaPipe Pose Landmarker and Face Landmarker.
It estimates eye aspect ratio (EAR) and facial head orientation at
one-second intervals. Sustained eye closure supported by the configured
head-orientation range is classified as SLEEPING. Clearly open eyes
are classified as AWAKE. Insufficient evidence is marked UNCERTAIN.

The multi-person pipeline detects up to two people, associates faces with
pose-guided head crops, assigns person labels, and generates per-person
timeline CSVs, interval JSON, and summary JSON.

## Development-set results

Annotated samples: 163
True positives: 19
False positives: 0
False negatives: 1
True negatives: 143

- Precision: 1.000
- Recall: 0.950
- F1: 0.974
- Specificity: 1.000
- Accuracy: 0.994
- Decision coverage: 0.521

The main evaluation treats UNCERTAIN as not predicted sleeping. The
decided-only metrics exclude uncertain frames and must be reported with
their coverage.

## Per-video results

| Video | Predicted sleep intervals | Uncertain samples |
|---|---:|---:|
| awake_01 | 0 | 10 |
| awake_02 | 0 | 15 |
| awake_03 | 0 | 1 |
| awake_eyes_closed_01 | 0 | 19 |
| awake_looking_down_01 | 0 | 32 |
| sleep_01 | 1 | 1 |

The sleep interval detected in sleep_01 was approximately 3.0–21.2 seconds.
The initial annotation estimated sleep onset at 1.5 seconds; that boundary
has not yet been independently verified.

## Two-person smoke test

A synthetic side-by-side video produced two pose detections and two
face detections at all 21 sampled timestamps. The pipeline classified
Person B as sleeping from approximately 5.0 to 20.3 seconds and did not
produce a sleep interval for Person A.

This is a pipeline smoke test, not independent evidence of accuracy on
real footage containing two people.

## Limitations

- Only one positive sleep clip was evaluated.
- Thresholds were developed on the same clips used for evaluation.
- Decision coverage is low on several awake examples.
- Pose-center tracking and initial left-to-right identity assignment
  may fail if people cross, overlap, or leave the scene.
- A simulated sleeping pose is not physiological ground truth.

Treat all results as preliminary. Evaluate on unseen recordings before
making claims about real-world generalization.
