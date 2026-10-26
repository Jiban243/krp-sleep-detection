import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

from src.video_reader import inspect_video


CLASSIFIER_MODEL = "awake-or-sleep/1"


def extract_pose_boxes(result, frame_width, frame_height):
    """Create expanded person crops from MediaPipe pose landmarks."""
    detections = []

    for landmarks in result.pose_landmarks:
        points = []

        for landmark in landmarks:
            visibility = float(getattr(landmark, "visibility", 1.0))

            if visibility < 0.35:
                continue

            x = float(landmark.x)
            y = float(landmark.y)

            if not math.isfinite(x) or not math.isfinite(y):
                continue

            points.append((x * frame_width, y * frame_height))

        if len(points) < 8:
            continue

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]

        x1 = min(x_values)
        y1 = min(y_values)
        x2 = max(x_values)
        y2 = max(y_values)

        width = x2 - x1
        height = y2 - y1

        if width < 20 or height < 30:
            continue

        # Keep some surrounding scene context in the crop.
        pad_x = width * 0.20
        pad_y = height * 0.15

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(frame_width, int(x2 + pad_x))
        y2 = min(frame_height, int(y2 + pad_y))

        if x2 - x1 < 20 or y2 - y1 < 30:
            continue

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "visibility": sum(
                getattr(landmark, "visibility", 1.0)
                for landmark in landmarks
                if getattr(landmark, "visibility", 1.0) >= 0.35
            ) / len(points),
        })

    # Establish initial left-to-right ordering.
    return sorted(
        detections,
        key=lambda item: (
            item["bbox"][0] + item["bbox"][2]
        ) / 2,
    )


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )
    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_tracks(detections, tracks, sample_index, names, next_id):
    """Associate detections between sampled frames using box overlap/position."""
    pairs = []

    for track_id, track in tracks.items():
        if sample_index - track["last_frame"] > 2:
            continue

        old_box = track["bbox"]

        for index, detection in enumerate(detections):
            new_box = detection["bbox"]
            overlap = iou(old_box, new_box)

            old_center = (
                (old_box[0] + old_box[2]) / 2,
                (old_box[1] + old_box[3]) / 2,
            )
            new_center = (
                (new_box[0] + new_box[2]) / 2,
                (new_box[1] + new_box[3]) / 2,
            )

            old_diagonal = math.hypot(
                old_box[2] - old_box[0],
                old_box[3] - old_box[1],
            )

            scale = max(old_diagonal, 1.0)

            distance = math.hypot(
                old_center[0] - new_center[0],
                old_center[1] - new_center[1],
            ) / scale

            if overlap < 0.02 and distance > 0.70:
                continue

            score = overlap + max(0.0, 0.70 - distance) * 0.15
            pairs.append((score, track_id, index))

    pairs.sort(reverse=True)

    used_tracks = set()
    used_detections = set()

    for _, track_id, index in pairs:
        if track_id in used_tracks or index in used_detections:
            continue

        detections[index]["track_id"] = track_id
        used_tracks.add(track_id)
        used_detections.add(index)

    for index, detection in enumerate(detections):
        if index in used_detections:
            continue

        track_id = f"T{next_id:03d}"

        name = (
            names[next_id - 1]
            if next_id <= len(names)
            else f"Person {next_id}"
        )

        tracks[track_id] = {
            "name": name,
            "bbox": detection["bbox"],
            "last_frame": sample_index,
        }

        detection["track_id"] = track_id
        next_id += 1

    for detection in detections:
        track = tracks[detection["track_id"]]
        track["bbox"] = detection["bbox"]
        track["last_frame"] = sample_index

    return next_id


def classify_crop(client, crop_path, min_score, min_margin):
    """Return the class scores and a conservative state estimate."""
    response = client.infer(
        str(crop_path),
        model_id=CLASSIFIER_MODEL,
    )

    scores = {}
    predictions = response.get("predictions", {})

    if isinstance(predictions, dict):
        for label, value in predictions.items():
            if isinstance(value, dict):
                value = value.get("confidence")

            try:
                score = float(value)
            except (TypeError, ValueError):
                continue

            if math.isfinite(score):
                scores[str(label).strip().lower()] = score

    elif isinstance(predictions, list):
        for item in predictions:
            if not isinstance(item, dict):
                continue

            label = str(
                item.get("class", item.get("class_name", ""))
            ).strip().lower()

            try:
                score = float(item["confidence"])
            except (KeyError, TypeError, ValueError):
                continue

            if label and math.isfinite(score):
                scores[label] = score

    if not scores:
        return {
            "state": "UNCERTAIN",
            "top_class": None,
            "top_score": None,
            "margin": None,
            "scores": {},
        }

    ordered = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_class, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = top_score - second_score

    if top_class not in {"awake", "sleep", "sleeping"}:
        state = "UNCERTAIN"
    elif top_score < min_score or margin < min_margin:
        state = "UNCERTAIN"
    elif top_class in {"sleep", "sleeping"}:
        state = "SLEEPING"
    else:
        state = "AWAKE"

    return {
        "state": state,
        "top_class": top_class,
        "top_score": round(top_score, 4),
        "margin": round(margin, 4),
        "scores": {
            key: round(value, 4)
            for key, value in scores.items()
        },
    }


def smooth_states(rows, max_gap):
    """Confirm sleeping only after two consecutive sleep predictions."""
    by_track = defaultdict(list)

    for row in rows:
        by_track[row["track_id"]].append(row)

    for track_rows in by_track.values():
        track_rows.sort(key=lambda row: row["timestamp_seconds"])

        for row in track_rows:
            row["state"] = (
                "AWAKE"
                if row["raw_state"] == "AWAKE"
                else "UNCERTAIN"
            )

        index = 0

        while index < len(track_rows):
            if track_rows[index]["raw_state"] != "SLEEPING":
                index += 1
                continue

            end = index + 1

            while end < len(track_rows):
                previous = track_rows[end - 1]
                current = track_rows[end]

                gap = (
                    current["timestamp_seconds"]
                    - previous["timestamp_seconds"]
                )

                if (
                    current["raw_state"] != "SLEEPING"
                    or gap > max_gap
                ):
                    break

                end += 1

            if end - index >= 2:
                for position in range(index, end):
                    track_rows[position]["state"] = "SLEEPING"

            index = end

    return rows


def build_intervals(rows, duration, max_gap, sample_period):
    by_track = defaultdict(list)

    for row in rows:
        by_track[row["track_id"]].append(row)

    intervals = []

    for track_id, track_rows in by_track.items():
        track_rows.sort(key=lambda row: row["timestamp_seconds"])

        start = None
        previous_time = None
        name = track_rows[0]["name"]

        for row in track_rows:
            timestamp = row["timestamp_seconds"]

            if row["state"] == "SLEEPING":
                if start is None:
                    start = timestamp
                elif timestamp - previous_time > max_gap:
                    intervals.append({
                        "track_id": track_id,
                        "name": name,
                        "start_seconds": round(start, 3),
                        "end_seconds": round(
                            min(duration, previous_time + sample_period),
                            3,
                        ),
                    })
                    start = timestamp

                previous_time = timestamp

            elif start is not None:
                intervals.append({
                    "track_id": track_id,
                    "name": name,
                    "start_seconds": round(start, 3),
                    "end_seconds": round(
                        min(duration, previous_time + sample_period),
                        3,
                    ),
                })
                start = None
                previous_time = None

        if start is not None:
            intervals.append({
                "track_id": track_id,
                "name": name,
                "start_seconds": round(start, 3),
                "end_seconds": round(
                    min(duration, previous_time + sample_period),
                    3,
                ),
            })

    return intervals


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Use MediaPipe to crop detected people and a pretrained "
            "awake/sleep classifier to estimate their states."
        )
    )

    parser.add_argument("--video", required=True)
    parser.add_argument("--names", default="Person A,Person B")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-people", type=int, default=2)
    parser.add_argument("--min-class-score", type=float, default=0.55)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--output-dir", default="outputs/pretrained_pose_classifier")

    args = parser.parse_args()

    video_path = Path(args.video).expanduser()
    model_path = Path("models/pose_landmarker_lite.task")

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model not found: {model_path}")

    if args.sample_fps <= 0 or args.max_people < 1:
        parser.error("Sampling FPS and maximum people must be positive.")

    api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is missing from the current terminal session."
        )

    info = inspect_video(video_path)
    stride = max(1, round(info.fps / args.sample_fps))
    sample_period = stride / info.fps
    actual_fps = info.fps / stride

    names = [
        name.strip()
        for name in args.names.split(",")
        if name.strip()
    ]

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    ).configure(
        InferenceConfiguration(api_key_transport="header")
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    timeline_path = output_dir / f"{stem}_timeline.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    annotated_path = output_dir / f"{stem}_annotated.mp4"

    crop_path = output_dir / "_temporary_person_crop.jpg"

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path.resolve())
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=args.max_people,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_fps,
        (info.width, info.height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create {annotated_path}")

    rows = []
    tracks = {}
    source_index = 0
    sampled_index = 0
    next_id = 1

    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = capture.read()

                if not success:
                    break

                index = source_index
                source_index += 1

                if index % stride != 0:
                    continue

                timestamp = index / info.fps
                timestamp_ms = round(timestamp * 1000)
                sampled_index += 1

                height, width = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                result = landmarker.detect_for_video(
                    image,
                    timestamp_ms,
                )

                detections = extract_pose_boxes(
                    result,
                    width,
                    height,
                )

                next_id = match_tracks(
                    detections,
                    tracks,
                    sampled_index,
                    names,
                    next_id,
                )

                observed_ids = set()

                for detection in detections:
                    track_id = detection["track_id"]
                    track = tracks[track_id]
                    observed_ids.add(track_id)

                    x1, y1, x2, y2 = detection["bbox"]
                    crop = frame[y1:y2, x1:x2]

                    prediction = {
                        "state": "UNCERTAIN",
                        "top_class": None,
                        "top_score": None,
                        "margin": None,
                        "scores": {},
                    }

                    if crop.size and cv2.imwrite(str(crop_path), crop):
                        try:
                            prediction = classify_crop(
                                client,
                                crop_path,
                                args.min_class_score,
                                args.min_margin,
                            )
                        except Exception as error:
                            print(
                                f"Classification failed at "
                                f"{timestamp:.2f}s for {track_id}: {error}"
                            )

                    rows.append({
                        "timestamp_seconds": round(timestamp, 3),
                        "track_id": track_id,
                        "name": track["name"],
                        "raw_state": prediction["state"],
                        "state": prediction["state"],
                        "top_class": prediction["top_class"],
                        "top_score": prediction["top_score"],
                        "score_margin": prediction["margin"],
                        "awake_score": prediction["scores"].get("awake"),
                        "sleep_score": prediction["scores"].get("sleep"),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    })

                    state = prediction["state"]
                    color = (
                        (0, 0, 255) if state == "SLEEPING"
                        else (0, 200, 0) if state == "AWAKE"
                        else (0, 200, 255)
                    )

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color, 2
                    )

                    cv2.putText(
                        frame,
                        f"{track['name']}: {state}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                # Brief detection gaps remain uncertain.
                for track_id, track in tracks.items():
                    if (
                        track_id not in observed_ids
                        and sampled_index - track["last_frame"] <= 2
                    ):
                        rows.append({
                            "timestamp_seconds": round(timestamp, 3),
                            "track_id": track_id,
                            "name": track["name"],
                            "raw_state": "UNCERTAIN",
                            "state": "UNCERTAIN",
                            "top_class": None,
                            "top_score": None,
                            "score_margin": None,
                            "awake_score": None,
                            "sleep_score": None,
                            "x1": None,
                            "y1": None,
                            "x2": None,
                            "y2": None,
                        })

                cv2.putText(
                    frame,
                    f"Time: {timestamp:.2f}s",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                writer.write(frame)

    finally:
        capture.release()
        writer.release()
        crop_path.unlink(missing_ok=True)

    rows = smooth_states(rows, max_gap=sample_period * 1.6)

    rows.sort(
        key=lambda row: (
            row["timestamp_seconds"],
            row["track_id"],
        )
    )

    columns = [
        "timestamp_seconds",
        "track_id",
        "name",
        "raw_state",
        "state",
        "top_class",
        "top_score",
        "score_margin",
        "awake_score",
        "sleep_score",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    with timeline_path.open("w", newline="") as file:
        output = csv.DictWriter(file, fieldnames=columns)
        output.writeheader()
        output.writerows(rows)

    intervals = build_intervals(
        rows,
        info.duration_seconds,
        max_gap=sample_period * 1.6,
        sample_period=sample_period,
    )

    summary = {
        "video": str(video_path.resolve()),
        "duration_seconds": round(info.duration_seconds, 3),
        "sampling_fps": round(actual_fps, 3),
        "pose_model": "MediaPipe Pose Landmarker Lite",
        "sleep_classifier": CLASSIFIER_MODEL,
        "classification_threshold": {
            "min_class_score": args.min_class_score,
            "min_score_margin": args.min_margin,
        },
        "track_name_mapping": {
            track_id: track["name"]
            for track_id, track in tracks.items()
        },
        "sleep_intervals": intervals,
        "limitations": [
            "The classifier is pretrained and has not been fine-tuned on these recordings.",
            "The model's published validation score does not establish accuracy on these videos.",
            "The classifier receives detected-person crops rather than the whole frame.",
            "Tracking uses box overlap and position, not real-world face recognition.",
            "Names must be mapped to tracks; left-to-right first detection is only a prototype convention.",
            "UNCERTAIN means the classification scores were ambiguous or the person could not be reliably observed.",
        ],
    }

    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print("\nPretrained pose-crop classifier")
    print("-------------------------------")
    print(f"Video: {video_path.resolve()}")
    print(f"Sampled frames: {sampled_index}")
    print(f"Person tracks: {len(tracks)}")
    print(f"Rows written: {len(rows)}")
    print(f"Sleep intervals: {len(intervals)}")

    for interval in intervals:
        print(
            f"  {interval['name']} ({interval['track_id']}): "
            f"{interval['start_seconds']:.2f}s - "
            f"{interval['end_seconds']:.2f}s"
        )

    print(f"\nTimeline: {timeline_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Annotated video: {annotated_path.resolve()}")


if __name__ == "__main__":
    main()
