# Video-Based Sleep Detection

An experimental computer vision prototype that estimates sleep-like behavior from video, tracks up to two people, and reports per-person states and estimated sleep intervals.

The system uses MediaPipe Pose Landmarker and Face Landmarker, eye aspect ratio (EAR), facial head orientation, and temporal smoothing. It supports three states: `AWAKE`, `SLEEPING`, and `UNCERTAIN`.

## Features

- Pose detection and experimental tracking for up to two people.
- Pose-guided face detection using enlarged head crops.
- Eye aspect ratio and head-orientation feature extraction.
- Temporal confirmation of sustained sleep-like behavior.
- Per-person timestamped CSV timelines.
- Sleep-interval JSON files and run-summary JSON files.
- Annotated MP4 generation for visual inspection.
- Evaluation against manually defined time intervals.

## Tech Stack

- Python 3.11
- OpenCV
- MediaPipe Tasks
- NumPy
- CSV and JSON for predictions and evaluation artifacts

## Quickstart

### 1. Requirements

Use Python 3.11 and ensure the included MediaPipe model assets exist:

```text
models/
├── pose_landmarker_lite.task
└── face_landmarker.task
```

### 2. Install dependencies

Create a virtual environment and install the pinned runtime dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-runtime.txt
```

### 3. Run the demonstration

Run the complete demonstration with one command:

```bash
./scripts/run_demo.sh
```

The demo uses the included synthetic sample video:

`data/demo/two_people_synthetic.mp4`

It runs sleep-like state detection and renders an annotated video. The generated timelines, intervals, summaries, and annotated MP4 are saved under `outputs/multi_person/`.

## Analyze Another Video

To analyze a video available on your machine, run:

```bash
python scripts/run_multi_person_sleep.py \
  --video /path/to/input_video.mp4 \
  --names "Person A,Person B"
```

Replace `/path/to/input_video.mp4` with the actual path to your video.

The `--names` argument assigns labels to tracked people. The current tracker uses initial left-to-right assignment and pose-center matching, so identity switches may occur if people cross, overlap, or leave and re-enter the scene.

Use the following command to view all available options:

```bash
python scripts/run_multi_person_sleep.py --help
```

## Operational Definition of Sleeping

The system estimates a **sleep-like visual state** rather than determining physiological sleep.

Video is sampled at one-second intervals. For each detected person, the pipeline extracts facial landmarks and calculates eye aspect ratio (EAR).

The current heuristic uses the following conditions:

- **AWAKE:** EAR is at least 0.16.
- **Sleep candidate:** EAR is at most 0.12 and the estimated head orientation falls within the configured ranges.
- **SLEEPING:** At least two consecutive sleep-candidate samples are observed.
- **UNCERTAIN:** The evidence is insufficient or does not satisfy the required conditions.

The configured head-orientation ranges are 10–23 degrees for rotation X and 1–10 degrees for rotation Z.

These thresholds were developed using the available recordings. They are experimental heuristics, not universal definitions of sleep.

## Pipeline

1. Sample frames from the input video.
2. Detect body poses and associate detections with person tracks.
3. Use pose landmarks to locate and enlarge each person's head crop.
4. Detect facial landmarks and extract eye and head-orientation features.
5. Classify samples as `AWAKE`, `SLEEPING`, or `UNCERTAIN`.
6. Confirm sustained sleep candidates and calculate intervals.
7. Save structured results and optionally render an annotated video.

## Output Files

For an input video named `sleep_01.mp4`, the pipeline produces:

| File | Description |
|---|---|
| `sleep_01_timeline.csv` | Per-person states, timestamps, track IDs, detection flags, EAR, head orientation, and classification evidence |
| `sleep_01_intervals.json` | Estimated sleep-like intervals with person labels, start and end times, and durations |
| `sleep_01_summary.json` | Video metadata, detection counts, per-person state counts, and detected intervals |

The files are written to `outputs/multi_person/` by default.

## Render an Annotated Video

After generating a timeline, run:

```bash
python scripts/render_multi_person_video.py \
  --video data/demo/two_people_synthetic.mp4
```

The renderer saves an annotated MP4 under `outputs/multi_person/`, showing person boxes, names, and predicted states.

Predictions are sampled at one-second intervals and held between samples. The video is a visualization of the existing predictions, not an independent classification at every frame.

## Evaluation

The manually specified reference intervals are stored in:

`evaluation/annotations/intervals.csv`

The evaluation script compares generated timelines with those annotations:

```bash
python scripts/evaluate_multi_person.py
```

The resulting metrics are written to:

`evaluation/multi_person_metrics.json`

To reproduce the full evaluation, generate timelines for all annotated source videos first. Those original recordings are not included in this repository; the included synthetic demo is intended for a quick smoke test.

### Preliminary Development-Set Results

The current annotated development set contains 163 sampled observations across six clips, including only one staged-sleep recording.

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

In the overall confusion matrix, `UNCERTAIN` samples are counted as not predicted sleeping. Decision coverage is the fraction of observations classified as either `AWAKE` or `SLEEPING`.

The perfect decided-only metrics exclude uncertain observations and must be interpreted alongside the 52.1% overall coverage.

These are preliminary development-set results, not independent test results. Thresholds were developed using the same recordings used for evaluation.

## Two-Person Smoke Test

A synthetic side-by-side video was created from separate awake and staged-sleep recordings.

At all 21 sampled timestamps, the pipeline detected two poses and two faces. It reported a sleep-like interval for Person B of approximately 5.0–20.3 seconds and did not produce a sleep interval for Person A.

This test demonstrates a basic multi-person processing path. It is **not independent validation on genuine footage containing two people together**.

## Limitations and Future Work

- The evaluation contains only one positive staged-sleep recording.
- Several awake recordings have low decision coverage, particularly the looking-down example.
- Eye closure and head posture are indirect visual cues and cannot establish physiological sleep.
- Sleep onset is not localized perfectly.
- Identity tracking may fail during crossings, occlusions, or re-entry.
- The synthetic two-person demonstration does not establish real-world accuracy.

With more time, the next priorities would be to collect a larger dataset with multiple subjects, lighting conditions, poses, and genuine two-person videos; verify interval annotations; evaluate on independent recordings; improve tracking and temporal classification; and measure both interval-level timing error and false-positive rates.

Longer term, the system could evolve into a privacy-conscious real-time monitoring tool with robust multi-person tracking, confidence-aware alerts, a dashboard for reviewing timelines, and exportable reports. On-device processing could reduce the need to transmit video. Optional contextual or wearable signals could help validate visual estimates. Any such system would require rigorous independent validation before being used for consequential decisions.

## Project Structure

```text
.
├── data/
│   └── demo/                  # Included synthetic demonstration video
├── docs/
│   ├── PROJECT_REPORT.md      # Detailed project report
│   └── SHORT_WRITEUP.md       # Concise assignment write-up
├── evaluation/
│   ├── annotations/           # Reference intervals
│   └── multi_person_metrics.json
├── models/                    # MediaPipe task model assets
├── outputs/
│   └── multi_person/          # Generated timelines and results
├── scripts/
│   ├── run_multi_person_sleep.py
│   ├── evaluate_multi_person.py
│   ├── render_multi_person_video.py
│   └── run_demo.sh
├── requirements-runtime.txt   # Pinned runtime dependencies
├── requirements.txt          # Full environment snapshot
└── README.md
```

## Additional Documentation

See [`docs/SHORT_WRITEUP.md`](docs/SHORT_WRITEUP.md) for the concise write-up covering the operational definition, signal selection, evaluation, failure modes, next steps, and long-term vision.

See [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) for additional methodology, results, and limitations.


