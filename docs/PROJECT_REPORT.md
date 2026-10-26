# Video-Based Sleep Detection

## 1. Project overview

This prototype analyzes videos to identify sleep-like behavior at
one-second intervals. It supports up to two detected people and produces
per-person state predictions and sleep intervals.

The implementation uses local MediaPipe Pose Landmarker and Face
Landmarker models. It does not require a hosted classification API.

## 2. Pipeline

1. Sample video frames at one-second intervals.
2. Detect body poses and associate detections with person tracks.
3. Estimate face and eye landmarks in pose-guided head crops.
4. Calculate eye aspect ratio (EAR) and facial head orientation.
5. Classify each sample as AWAKE, SLEEPING, or UNCERTAIN.
6. Confirm sleep-like states only when the required evidence persists.
7. Write per-person timelines, sleep intervals, and summary metrics.

## 3. State classification

The experimental classifier uses:
- EAR of at least 0.16 as evidence of AWAKE.
- EAR of at most 0.12 as evidence of eye closure.
- Head-orientation X range of 10–23 degrees.
- Head-orientation Z range of 1–10 degrees.
- At least two consecutive sleep-candidate samples to confirm sleep.

Samples that do not meet the required conditions remain UNCERTAIN.
These thresholds were developed on the available recordings and are
not universal physiological definitions of sleep.

## 4. Running the pipeline

Run a single-person analysis:

    python scripts/run_multi_person_sleep.py \
      --video data/raw/sleep_01.mp4 \
      --names "Person A"

Run the two-person synthetic smoke test:

    python scripts/run_multi_person_sleep.py \
      --video data/raw/two_people_synthetic.mp4 \
      --names "Person A,Person B"

Evaluate predictions against the configured annotation file:

    python scripts/evaluate_multi_person.py

Required local model assets:
- models/pose_landmarker_lite.task
- models/face_landmarker.task

Install project dependencies in the active virtual environment using:

    pip install -r requirements.txt

## 5. Outputs

For each input video, the multi-person script writes:

- `*_timeline.csv`: one row per tracked person and sampled timestamp.
  Includes person label, track ID, pose/face detection flags, EAR,
  head orientation, raw evidence state, and final state.
- `*_intervals.json`: detected sleep-like intervals with person label,
  track ID, start/end times, and interval duration.
- `*_summary.json`: video metadata, detection counts, per-person state
  counts, and detected intervals.

Evaluation metrics are written to:

    evaluation/multi_person_metrics.json

## 6. Preliminary development-set evaluation

The available annotated recordings contain 163 sampled observations,
including 20 annotated sleep observations from one staged-sleep clip.

| Metric | Result |
|---|---:|
| True positives | 19 |
| False positives | 0 |
| False negatives | 1 |
| True negatives | 143 |
| Precision | 1.000 |
| Recall | 0.950 |
| F1 | 0.974 |
| Specificity | 1.000 |
| Accuracy | 0.994 |
| Decision coverage | 52.1% |

UNCERTAIN samples are counted as not predicted sleeping in the overall
confusion matrix. Coverage is the fraction of samples classified as
AWAKE or SLEEPING rather than UNCERTAIN.

The decided-only F1 is 1.000, but it excludes uncertain samples and
must be interpreted together with its 52.1% overall coverage.

## 7. Two-person smoke test

A synthetic side-by-side video created from separate awake and staged-sleep
recordings produced two pose detections and two face detections at all
21 sampled timestamps.

The prototype reported a sleep-like interval for Person B from
approximately 5.0 to 20.3 seconds and no sleep interval for Person A.

This verifies a basic pipeline path, not accuracy on a genuine video
containing two people together.

## 8. Limitations

- The evaluation contains only one positive staged-sleep recording.
- Thresholds were developed using the same recordings used for evaluation.
- High overall accuracy is influenced by the predominance of awake samples.
- Decision coverage is low for some awake recordings, particularly
  when the person looks down or appears at an angle.
- The estimated sleep interval begins later than the initial annotated
  onset; the transition is not perfectly localized.
- Initial left-to-right identity assignment and pose-center matching
  may fail when people cross, overlap, disappear, or re-enter the scene.
- A staged pose is not physiological ground truth.
- Real-world accuracy requires an independent test set with genuine
  two-person footage and verified time-interval annotations.

## 9. Interpretation

This is an experimental sleep-like-state detector with explicit
uncertainty handling. Results are preliminary development-set metrics
and should not be presented as evidence of generalization to unseen
people, environments, or real-world sleeping behavior.
