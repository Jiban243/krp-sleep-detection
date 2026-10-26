# Video-Based Sleep Detection

An experimental computer vision prototype that estimates sleep-like
behavior from video and reports per-person states and time intervals.
It uses MediaPipe Pose Landmarker, Face Landmarker, eye aspect ratio
(EAR), facial head orientation, and temporal smoothing.

The system supports the states `AWAKE`, `SLEEPING`, and `UNCERTAIN`,
with experimental support for tracking up to two people.

## Features

- Video sampling at configurable intervals.
- Pose detection for up to two people.
- Pose-guided face detection using enlarged head crops.
- Eye aspect ratio and head-orientation features.
- Temporal confirmation of sustained sleep-like behavior.
- Per-person CSV timelines and JSON sleep intervals.
- JSON run summaries and evaluation metrics.
- Annotated MP4 output for visual inspection.

## Tech Stack

Python 3.11, OpenCV, MediaPipe Tasks, NumPy, CSV, and JSON.

## Setup

Create and activate a virtual environment:

    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

The following local model assets are required:

    models/pose_landmarker_lite.task
    models/face_landmarker.task

Ensure the model files are available at these paths before running
the scripts. Original input videos are not included in the repository.

## Run Sleep Detection

Analyze a single-person video:

    python scripts/run_multi_person_sleep.py \
      --video data/raw/sleep_01.mp4 \
      --names "Person A"

Analyze a video containing two people:

    python scripts/run_multi_person_sleep.py \
      --video /path/to/two_people.mp4 \
      --names "Person A,Person B"

By default, the results are written to `outputs/multi_person/`.

Useful options include `--output-dir`, `--sample-step`,
`--closed-threshold`, `--open-threshold`, and `--min-duration`.
Use `python scripts/run_multi_person_sleep.py --help` for all options.

## Output Files

For an input video named `sleep_01.mp4`, the system generates:

- `sleep_01_timeline.csv` — sampled states, person labels, track IDs,
  detection flags, EAR, head orientation, and final decisions.
- `sleep_01_intervals.json` — estimated sleep-like intervals, including
  person name, start time, end time, and duration.
- `sleep_01_summary.json` — video metadata, detection counts, state
  counts, and detected intervals.

## Render an Annotated Video

After generating a timeline, run:

    python scripts/render_multi_person_video.py \
      --video data/raw/sleep_01.mp4

The rendered video is saved in `outputs/multi_person/` by default.
It displays person boxes, labels, and predicted states. Predictions
are sampled at one-second intervals and held between samples.

## Evaluation

Ground-truth intervals are stored in:

    evaluation/annotations/intervals.csv

After generating prediction timelines for the annotated videos, run:

    python scripts/evaluate_multi_person.py

The metrics are written to `evaluation/multi_person_metrics.json`.

### Preliminary Development-Set Results

| Metric | Result |
|---|---:|
| True positives | 19 |
| False positives | 0 |
| False negatives | 1 |
| True negatives | 143 |
| Precision | 1.000 |
| Recall | 0.950 |
| F1 score | 0.974 |
| Specificity | 1.000 |
| Accuracy | 0.994 |
| Decision coverage | 52.1% |

These metrics are from 163 sampled observations, including only one
staged-sleep recording. In the overall confusion matrix, `UNCERTAIN`
samples are counted as not predicted sleeping. Decision coverage is
the proportion classified as either `AWAKE` or `SLEEPING`.

The perfect decided-only metrics exclude uncertain observations and
must be interpreted together with the 52.1% overall coverage.

## Two-Person Smoke Test

A synthetic side-by-side video assembled from separate awake and
staged-sleep clips produced two pose and face detections at all
21 sampled timestamps. The pipeline reported a sleep-like interval
for Person B of approximately 5.0–20.3 seconds.

This is a pipeline smoke test, not validation on genuine footage
containing two people together.

## Limitations

- Only one positive staged-sleep recording was evaluated.
- Thresholds were developed using the same clips used for evaluation.
- Some awake recordings have low decision coverage.
- Eye closure and head posture are indirect indicators, not proof of sleep.
- Sleep onset is not localized perfectly.
- Tracking may fail when people cross, overlap, or leave the frame.
- The synthetic two-person test does not establish real-world accuracy.

The metrics are preliminary development-set results and do not
establish generalization to unseen people or environments.

## Project Structure

    docs/                   Project report
    evaluation/             Annotations and evaluation results
    models/                 MediaPipe task model assets
    outputs/multi_person/   Generated timelines and results
    scripts/                Detection, evaluation, and rendering scripts
    data/raw/               Local input videos (not committed)
    requirements.txt        Python dependencies

## Reproducible Demo Quickstart



Requirements: Python 3.11 and the MediaPipe model assets in `models/`.



Create the environment and install the pinned runtime dependencies:



```bash

python3.11 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install -r requirements-runtime.txt

./scripts/run_demo.sh

```



The demo uses the included synthetic video at `data/demo/two_people_synthetic.mp4` and generates CSV timelines, JSON intervals, summaries, and an annotated MP4 under `outputs/multi_person/`.



For another local video, run:



```bash

python scripts/run_multi_person_sleep.py --video /path/to/input_video.mp4 --names "Person A,Person B"

```



Replace `/path/to/input_video.mp4` with the actual path to your video.
