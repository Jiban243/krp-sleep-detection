import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

from src.video_reader import inspect_video


PERSON_MODEL = "guard-sleep-detection/1"
SLEEP_CLASSIFIER = "awake-or-sleep/1"


def parse_prediction_box(prediction, frame_width, frame_height):
    """Parse a center-based detection box and keep person classes only."""
    label = str(prediction.get("class", "")).strip().lower()

    if label != "person":
        return None

    try:
        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
        confidence = float(prediction.get("confidence", 0.0))
    except (KeyError, TypeError, ValueError):
        return None

    values = (x, y, width, height, confidence)

    if not all(math.isfinite(v) for v in values):
        return None

    if width <= 0 or height <= 0:
        return None

    x1 = max(0, int(x - width / 2))
    y1 = max(0, int(y - height / 2))
    x2 = min(frame_width, int(x + width / 2))
    y2 = min(frame_height, int(y + height / 2))

    if x2 - x1 < 10 or y2 - y1 < 20:
        return None

    return {
        "bbox": (x1, y1, x2, y2),
        "confidence": confidence,
    }


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def suppress_duplicate_boxes(detections, threshold=0.60):
    """Simple non-maximum suppression for overlapping person boxes."""
    kept = []

    detections = sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    for candidate in detections:
        if all(
            iou(candidate["bbox"], existing["bbox"]) < threshold
            for existing in kept
        ):
            kept.append(candidate)

    return sorted(
        kept,
        key=lambda item: (item["bbox"][0] + item["bbox"][2]) / 2,
    )


def extract_person_boxes(response, width, height, min_confidence):
    predictions = response.get("predictions", [])

    detections = []

    for prediction in predictions:
        parsed = parse_prediction_box(prediction, width, height)

        if parsed is None:
            continue

        if parsed["confidence"] >= min_confidence:
            detections.append(parsed)

    return suppress_duplicate_boxes(detections)


def parse_classification(response):
    """Handle the usual Roboflow single-label classification response."""
    scores = {}

    predictions = response.get("predictions", [])

    if isinstance(predictions, list):
        for item in predictions:
            if not isinstance(item, dict):
                continue

            label = str(item.get("class", "")).strip().lower()

            try:
                score = float(item["confidence"])
            except (KeyError, TypeError, ValueError):
                continue

            if math.isfinite(score):
                scores[label] = score

    elif isinstance(predictions, dict):
        for label, value in predictions.items():
            if isinstance(value, dict):
                value = value.get("confidence")

            try:
                score = float(value)
            except (TypeError, ValueError):
                continue

            if math.isfinite(score):
                scores[str(label).lower()] = score

    if scores:
        label = max(scores, key=scores.get)
        confidence = scores[label]
    else:
        label = str(response.get("top", "")).strip().lower()

        try:
            confidence = float(response.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if label:
            scores[label] = confidence

    if label not in {"awake", "sleep"}:
        return "UNCERTAIN", confidence, scores

    return label.upper(), confidence, scores


def track_detections(detections, tracks, frame_index, names, next_id):
    """Associate person boxes using IoU and center distance."""
    frame_pairs = []

    for track_id, track in tracks.items():
        if frame_index - track["last_frame"] > 2:
            continue

        for index, detection in enumerate(detections):
            box_a = track["bbox"]
            box_b = detection["bbox"]

            overlap = iou(box_a, box_b)

            center_a = (
                (box_a[0] + box_a[2]) / 2,
                (box_a[1] + box_a[3]) / 2,
            )
            center_b = (
                (box_b[0] + box_b[2]) / 2,
                (box_b[1] + box_b[3]) / 2,
            )

            if overlap < 0.03:
                scale = max(
                    1.0,
                    math.hypot(
                        box_a[2] - box_a[0],
                        box_a[3] - box_a[1],
                    ),
                )
                distance = math.hypot(
                    center_a[0] - center_b[0],
                    center_a[1] - center_b[1],
                ) / scale

                if distance > 0.60:
                    continue

            score = overlap
            frame_pairs.append((score, track_id, index))

    frame_pairs.sort(reverse=True)

    matched_tracks = set()
    matched_detections = set()

    for _, track_id, index in frame_pairs:
        if track_id in matched_tracks or index in matched_detections:
            continue

        detections[index]["track_id"] = track_id
        matched_tracks.add(track_id)
        matched_detections.add(index)

    # New identities are assigned in left-to-right order.
    for index, detection in enumerate(detections):
        if index in matched_detections:
            continue

        track_id = f"T{next_id:03d}"

        if next_id <= len(names):
            name = names[next_id - 1]
        else:
            name = f"Person {next_id}"

        tracks[track_id] = {
            "name": name,
            "bbox": detection["bbox"],
            "last_frame": frame_index,
        }

        detection["track_id"] = track_id
        next_id += 1

    for detection in detections:
        track = tracks[detection["track_id"]]
        track["bbox"] = detection["bbox"]
        track["last_frame"] = frame_index

    return next_id


def smooth_states(rows):
    """Confirm sleep only if two adjacent sampled observations agree."""
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
            if track_rows[index]["raw_state"] != "SLEEP":
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
                    current["raw_state"] != "SLEEP"
                    or gap > 1.6
                ):
                    break

                end += 1

            if end - index >= 2:
                for position in range(index, end):
                    track_rows[position]["state"] = "SLEEPING"

            index = end

    return rows


def make_intervals(rows, duration, sample_period):
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

            continues = (
                start is not None
                and previous_time is not None
                and timestamp - previous_time <= 1.6 * sample_period
            )

            if row["state"] == "SLEEPING":
                if start is None:
                    start = timestamp
                elif not continues:
                    intervals.append({
                        "track_id": track_id,
                        "name": name,
                        "start_seconds": round(start, 3),
                        "end_seconds": round(
                            min(duration, previous_time + sample_period), 3
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
                        min(duration, previous_time + sample_period), 3
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
                    min(duration, previous_time + sample_period), 3
                ),
            })

    return intervals


def main():
    parser = argparse.ArgumentParser(
        description="Person-crop classification using a pretrained awake/sleep model."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--names", default="Person A,Person B")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument("--class-confidence", type=float, default=0.55)
    parser.add_argument("--output-dir", default="outputs/pretrained_classifier")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive")

    api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is missing from the current terminal session."
        )

    info = inspect_video(video_path)
    stride = max(1, round(info.fps / args.sample_fps))
    sample_period = stride / info.fps
    actual_fps = info.fps / stride

    names = [n.strip() for n in args.names.split(",") if n.strip()]

    detector = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    ).configure(
        InferenceConfiguration(api_key_transport="header")
    )

    classifier = InferenceHTTPClient(
        api_url="https://classify.roboflow.com",
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

    frame_path = output_dir / "_frame.jpg"
    crop_path = output_dir / "_person_crop.jpg"

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
    frame_number = 0
    sampled_frame = 0
    next_id = 1

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            index = frame_number
            frame_number += 1

            if index % stride != 0:
                continue

            timestamp = index / info.fps
            height, width = frame.shape[:2]
            sampled_frame += 1

            detections = []

            if cv2.imwrite(str(frame_path), frame):
                response = detector.infer(
                    str(frame_path),
                    model_id=PERSON_MODEL,
                )

                detections = extract_person_boxes(
                    response,
                    width,
                    height,
                    args.person_confidence,
                )

            next_id = track_detections(
                detections,
                tracks,
                sampled_frame,
                names,
                next_id,
            )

            observed_ids = set()

            for detection in detections:
                track_id = detection["track_id"]
                track = tracks[track_id]
                observed_ids.add(track_id)

                x1, y1, x2, y2 = detection["bbox"]

                # Keep a small margin around the detected person.
                margin_x = int((x2 - x1) * 0.06)
                margin_y = int((y2 - y1) * 0.06)

                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(width, x2 + margin_x)
                y2 = min(height, y2 + margin_y)

                crop = frame[y1:y2, x1:x2]

                raw_state = "UNCERTAIN"
                class_confidence = None
                awake_score = None
                sleep_score = None

                if crop.size and cv2.imwrite(str(crop_path), crop):
                    try:
                        prediction = classifier.infer(
                            str(crop_path),
                            model_id=SLEEP_CLASSIFIER,
                        )

                        label, confidence, scores = parse_classification(
                            prediction
                        )

                        sleep_score = scores.get("sleep")
                        awake_score = scores.get("awake")
                        class_confidence = confidence

                        if confidence >= args.class_confidence:
                            raw_state = (
                                "SLEEP" if label == "SLEEPING"
                                else "AWAKE" if label == "AWAKE"
                                else "UNCERTAIN"
                            )
                    except Exception as error:
                        print(
                            f"Classification failed at "
                            f"{timestamp:.2f}s for {track_id}: {error}"
                        )

                row = {
                    "timestamp_seconds": round(timestamp, 3),
                    "track_id": track_id,
                    "name": track["name"],
                    "raw_state": raw_state,
                    "state": raw_state,
                    "class_confidence": (
                        round(class_confidence, 4)
                        if class_confidence is not None else None
                    ),
                    "awake_score": (
                        round(awake_score, 4)
                        if awake_score is not None else None
                    ),
                    "sleep_score": (
                        round(sleep_score, 4)
                        if sleep_score is not None else None
                    ),
                    "person_detection_confidence": round(
                        detection["confidence"], 4
                    ),
                    "x1": detection["bbox"][0],
                    "y1": detection["bbox"][1],
                    "x2": detection["bbox"][2],
                    "y2": detection["bbox"][3],
                }
                rows.append(row)

                color = (
                    (0, 0, 255)
                    if raw_state == "SLEEP"
                    else (0, 200, 0)
                    if raw_state == "AWAKE"
                    else (0, 200, 255)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                cv2.putText(
                    frame,
                    f"{track['name']} {raw_state} "
                    f"{class_confidence if class_confidence is not None else 0:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            # Missing person observations remain uncertain for up to two samples.
            for track_id, track in tracks.items():
                if (
                    track_id not in observed_ids
                    and sampled_frame - track["last_frame"] <= 2
                ):
                    rows.append({
                        "timestamp_seconds": round(timestamp, 3),
                        "track_id": track_id,
                        "name": track["name"],
                        "raw_state": "UNCERTAIN",
                        "state": "UNCERTAIN",
                        "class_confidence": None,
                        "awake_score": None,
                        "sleep_score": None,
                        "person_detection_confidence": None,
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
        frame_path.unlink(missing_ok=True)
        crop_path.unlink(missing_ok=True)

    rows = smooth_states(rows)

    columns = [
        "timestamp_seconds",
        "track_id",
        "name",
        "raw_state",
        "state",
        "class_confidence",
        "awake_score",
        "sleep_score",
        "person_detection_confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    rows.sort(
        key=lambda row: (
            row["timestamp_seconds"],
            row["track_id"],
        )
    )

    with timeline_path.open("w", newline="") as file:
        output = csv.DictWriter(file, fieldnames=columns)
        output.writeheader()
        output.writerows(rows)

    intervals = make_intervals(
        rows,
        info.duration_seconds,
        sample_period,
    )

    summary = {
        "video": str(video_path.resolve()),
        "duration_seconds": round(info.duration_seconds, 3),
        "sampling_fps": round(actual_fps, 3),
        "person_detector": PERSON_MODEL,
        "sleep_classifier": SLEEP_CLASSIFIER,
        "class_confidence_threshold": args.class_confidence,
        "sleep_intervals": intervals,
        "track_name_mapping": {
            track_id: track["name"]
            for track_id, track in tracks.items()
        },
        "limitations": [
            "The classification model is pretrained; no fine-tuning was performed.",
            "The model's published metrics are not a guarantee of performance on these recordings.",
            "Person tracks use a simple IoU and center-distance association, not face recognition.",
            "Names follow track creation order and require a known mapping for reliable real-person identity.",
            "UNCERTAIN means evidence is insufficient; it does not mean awake.",
        ],
    }

    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print("\nPretrained classification pipeline")
    print("----------------------------------")
    print(f"Video: {video_path.resolve()}")
    print(f"Sampled frames: {sampled_frame}")
    print(f"Person tracks: {len(tracks)}")
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
