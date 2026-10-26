# Video-Based Sleep Detection — Short Write-up

## 1. Operational Definition of “Sleeping”

I defined sleeping operationally as a sustained, sleep-like visual state,
not as a physiological diagnosis. The video is sampled at one-second
intervals. For each tracked person, eye aspect ratio (EAR) is calculated
from facial landmarks. A sample is considered awake when EAR is at least
0.16. Eye closure, defined as EAR at most 0.12, is only a sleep candidate
when head orientation also falls within configured ranges (rotation X:
10–23 degrees; rotation Z: 1–10 degrees). At least two consecutive
sleep-candidate samples are required before the final state becomes
SLEEPING. Samples with conflicting or insufficient evidence are labeled
UNCERTAIN.

The thresholds are heuristic and were chosen by inspecting a staged-sleep
clip and challenging awake examples. Combining sustained eye closure with
head orientation is intended to reduce false detections from ordinary
blinks, deliberate eye closure, and looking down. These visual criteria
indicate sleep-like behavior; they do not prove that someone is asleep.

## 2. Signals and Why They Were Chosen

The system uses MediaPipe Pose Landmarker to detect bodies and maintain
experimental person tracks. Pose landmarks guide an enlarged head crop
for face detection, which was more reliable than searching for faces in
the full side-by-side frame. MediaPipe Face Landmarker supplies eye
landmarks for EAR and facial transformation matrices for approximate
head orientation.

EAR is an inexpensive, interpretable signal for sustained eye closure.
Head orientation adds contextual evidence that can help distinguish
eye closure while upright from a sustained head posture associated with
the staged-sleep example. Temporal smoothing prevents a single closed-eye
sample from immediately becoming a sleep interval. The UNCERTAIN state
is intentional: the system abstains when its evidence is not strong
enough instead of forcing every frame into awake or sleeping.

## 3. Evaluation and Comparison

I compared the predicted states at sampled timestamps against manually
specified time intervals in `evaluation/annotations/intervals.csv`.
The development set contains 163 observations across six clips: five
awake examples, including deliberate eye closure and looking down, and
one staged-sleep recording. The final heuristic yielded 19 true positives,
zero false positives, one false negative, and 143 true negatives.
Precision was 1.000, recall 0.950, F1 0.974, specificity 1.000, and
accuracy 0.994. Decision coverage was only 52.1%.

UNCERTAIN observations count as not predicted sleeping in the overall
confusion matrix. The decided-only F1 is 1.000 but excludes uncertain
frames, so it should not be read as performance over the entire video.
The two-person pipeline was also smoke-tested on a synthetic side-by-side
video assembled from separate awake and staged-sleep clips. It detected
two poses and two faces at all 21 sampled timestamps and reported a
sleep-like interval for Person B. This does not constitute validation on
genuine footage containing two people together.

## 4. Failure Modes and Next Steps

The final heuristic produced no false positives on the current labeled
awake clips, but this small result does not establish a low false-positive
rate on unseen data. An earlier eye-only baseline produced false positives
on awake examples; adding head-orientation constraints reduced those
errors on the same development clips. The main observed weakness is
abstention: coverage is low on some angled or looking-down recordings.
The onset in the staged-sleep clip is also delayed; the first confirmed
sleep sample occurs at about 3 seconds, while the initial annotation
estimated an onset around 1.5 seconds. That boundary needs more precise
frame-level annotation.

With more time, I would collect a larger, diverse dataset with multiple
subjects, camera angles, lighting conditions, occlusions, and real
two-person videos. I would create independent train, validation, and
test splits, double-check interval annotations, measure event-level
timing error, and tune thresholds only on training/validation data.
For tracking, I would compare the current pose-center association with a
more robust multi-object tracker and explicit identity re-association.

## 5. Long-Term Vision

The long-term goal is a privacy-conscious, real-time system that tracks
each person separately and produces interpretable timelines, confidence
scores, sleep-like intervals, and optional alerts. Useful additions
include temporal models trained on labeled sequences, better tracking
through crossings and occlusions, explicit low-quality-frame detection,
and a dashboard for reviewing annotated clips and exporting reports.
On-device processing would reduce the need to send video to a server.
Optional contextual or wearable signals, such as actigraphy, could help
validate or augment visual estimates where appropriate. Any such system
would still require rigorous independent validation; camera-only visual
cues should not be presented as a clinical measurement of sleep.
