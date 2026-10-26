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


MODEL_ID = "guard-sleep-detection/1"
API_URL = "https://serverless.roboflow.com"


def parse_box(prediction):
    """Convert a Roboflow center-based bounding box to x1, y1, x2, y2."""
    try:
        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
        confidence = float(prediction.get("confidence", 0.0))
    except (TypeError, ValueError, KeyError):
        return None

    if not all(
        math.isfinite(v)
        for v in (x, y, width, height, confidence)
    ):
        return None

    if width <= 0 or height <= 0:
        return None

    bbox = (
        x - width / 2,
        y - height / 2,
        x + width / 2,
        y + height / 2,
    )

    return {
        "bbox": bbox,
        "confidence": confidence,
        "class": str(
            prediction.get("class", prediction.get("class_name", ""))
        ).strip().lower(),
    }


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(
        0.0, box[3] - box[1]
    )


def intersection_area(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def box_iou(a, b):
    intersection = intersection_area(a, b)
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def box_overlap_ratio(a, b):
    """Intersection divided by the area of the smaller box."""
    denominator = min(box_area(a), box_area(b))
    return (
        intersection_area(a, b) / denominator
        if denominator > 0
        else 0.0
    )


def make_candidates(response, confidence_threshold):
    """Associate sleeping detections with person detections."""
    if not isinstance(response, dict):
        return []

    predictions = response.get("predictions", [])

    persons = []
    sleeping_boxes = []

    for prediction in predictions:
        parsed = parse_box(prediction)

        if parsed is None:
            continue

        if parsed["confidence"] < confidence_threshold:
            continue

        label = parsed["class"]

        if label == "person" or label.startswith("person_"):
            persons.append(parsed)
        elif "sleep" in label:
            sleeping_boxes.append(parsed)

    # Match each sleeping detection to the best overlapping person.
    sleep_matches = defaultdict(list)
    matched_sleep_indices = set()

    for sleep_index, sleeping in enumerate(sleeping_boxes):
        best_person_index = None
        best_overlap = 0.0

        for person_index, person in enumerate(persons):
            overlap = max(
                box_iou(person["bbox"], sleeping["bbox"]),
                box_overlap_ratio(person["bbox"], sleeping["bbox"]),
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_person_index = person_index

        if best_person_index is not None and best_overlap >= 0.25:
            sleep_matches[best_person_index].append(sleeping)
            matched_sleep_indices.add(sleep_index)

    candidates = []

    for person_index, person in enumerate(persons):
        matches = sleep_matches.get(person_index, [])

        if matches:
            best_sleep_confidence = max(
                detection["confidence"] for detection in matches
            )
            state = "SLEEPING"
            confidence = best_sleep_confidence
            detection_class = "sleeping"
        else:
            # This means the model did not label the person as sleeping;
            # it is not independent proof that the person is awake.
            state = "AWAKE"
            confidence = person["confidence"]
            detection_class = "person"

        candidates.append(
            {
                "bbox": person["bbox"],
                "raw_state": state,
                "confidence": confidence,
                "detection_class": detection_class,
            }
        )

    # Retain sleeping detections not paired with a person box.
    for sleep_index, sleeping in enumerate(sleeping_boxes):
        if sleep_index in matched_sleep_indices:
            continue

        candidates.append(
            {
                "bbox": sleeping["bbox"],
                "raw_state": "SLEEPING",
                "confidence": sleeping["confidence"],
                "detection_class": "sleeping",
            }
        )

    return sorted(
        candidates,
        key=lambda item: (
            item["bbox"][0] + item["bbox"][2]
        ) / 2,
    )


def match_score(track_box, candidate_box, frame_width, frame_height):
    overlap = box_iou(track_box, candidate_box)

    ax = (track_box[0] + track_box[2]) / 2
    ay = (track_box[1] + track_box[3]) / 2
    bx = (candidate_box[0] + candidate_box[2]) / 2
    by = (candidate_box[1] + candidate_box[3]) / 2

    diagonal = math.hypot(frame_width, frame_height)
    distance = (
        math.hypot(ax - bx, ay - by) / diagonal
        if diagonal > 0
        else 1.0
    )

    if overlap < 0.05 and distance > 0.15:
        return None

    return overlap + max(0.0, 0.15 - distance)


def assign_tracks(
    candidates,
    tracks,
    frame_number,
    frame_width,
    frame_height,
    names,
    next_track_id,
    max_age=2,
):
    """Greedy IoU/center-distance association for a mostly static camera."""
    pairs = []

    for track_id, track in tracks.items():
        if frame_number - track["last_frame"] > max_age:
            continue

        for candidate_index, candidate in enumerate(candidates):
            score = match_score(
                track["bbox"],
                candidate["bbox"],
                frame_width,
                frame_height,
            )

            if score is not None:
                pairs.append(
                    (score, track_id, candidate_index)
                )

    pairs.sort(reverse=True)

    matched_tracks = set()
    matched_candidates = set()

    for _, track_id, candidate_index in pairs:
        if track_id in matched_tracks:
            continue
        if candidate_index in matched_candidates:
            continue

        candidates[candidate_index]["track_id"] = track_id
        matched_tracks.add(track_id)
        matched_candidates.add(candidate_index)

    # New tracks are named left-to-right, in order of appearance.
    for candidate_index, candidate in enumerate(candidates):
        if candidate_index in matched_candidates:
            continue

        track_id = f"T{next_track_id:03d}"
        name_index = next_track_id - 1

        if name_index < len(names):
            name = names[name_index]
        else:
            name = f"Person {next_track_id}"

        tracks[track_id] = {
            "track_id": track_id,
            "name": name,
            "bbox": candidate["bbox"],
            "last_frame": frame_number,
            "last_timestamp": None,
        }

        candidate["track_id"] = track_id
        next_track_id += 1

    return matched_tracks | {
        candidate["track_id"] for candidate in candidates
    }, next_track_id


def smooth_states(rows, sample_period):
    """Require two nearby sleeping detections before confirming an interval."""
    by_track = defaultdict(list)

    for row in rows:
        by_track[row["track_id"]].append(row)

    for track_rows in by_track.values():
        track_rows.sort(key=lambda row: row["timestamp_seconds"])

        for index, row in enumerate(track_rows):
            if row["raw_state"] != "SLEEPING":
                row["state"] = row["raw_state"]
                continue

            sustained = False

            for other_index in (index - 1, index + 1):
                if not 0 <= other_index < len(track_rows):
                    continue

                other = track_rows[other_index]

                if other["raw_state"] != "SLEEPING":
                    continue

                gap = abs(
                    other["timestamp_seconds"]
                    - row["timestamp_seconds"]
                )

                if gap <= sample_period * 1.6:
                    sustained = True
                    break

            row["state"] = (
                "SLEEPING" if sustained else "UNCERTAIN"
            )

    return rows


def summarize_intervals(rows, duration, sample_period):
    by_track = defaultdict(list)

    for row in rows:
        by_track[row["track_id"]].append(row)

    intervals = []

    for track_id, track_rows in by_track.items():
        track_rows.sort(key=lambda row: row["timestamp_seconds"])
        current_start = None
        previous_time = None
        current_name = track_rows[0]["name"]

        for row in track_rows:
            timestamp = row["timestamp_seconds"]

            continues = (
                current_start is not None
                and previous_time is not None
                and timestamp - previous_time <= sample_period * 1.6
            )

            if row["state"] == "SLEEPING":
                if current_start is None:
                    current_start = timestamp
                elif not continues:
                    intervals.append(
                        {
                            "track_id": track_id,
                            "name": current_name,
                            "start_seconds": round(current_start, 3),
                            "end_seconds": round(
                                min(duration, previous_time + sample_period),
                                3,
                            ),
                        }
                    )
                    current_start = timestamp

                previous_time = timestamp

            elif current_start is not None:
                intervals.append(
                    {
                        "track_id": track_id,
                        "name": current_name,
                        "start_seconds": round(current_start, 3),
                        "end_seconds": round(
                            min(duration, previous_time + sample_period),
                            3,
                        ),
                    }
                )
                current_start = None
                previous_time = None

        if current_start is not None and previous_time is not None:
            intervals.append(
                {
                    "track_id": track_id,
                    "name": current_name,
                    "start_seconds": round(current_start, 3),
                    "end_seconds": round(
                        min(duration, previous_time + sample_period),
                        3,
                    ),
                }
            )

    return intervals


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pretrained sleep detection using Roboflow's "
            "Guard Sleep Detection model."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--names",
        default="Person A,Person B",
        help="Names in left-to-right order on first joint appearance.",
    )
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--output-dir", default="outputs/pretrained")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive")

    api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is missing. Set it in the current shell "
            "before running this script. Do not store the key in Git."
        )

    info = inspect_video(video_path)
    stride = max(1, round(info.fps / args.sample_fps))
    actual_sample_fps = info.fps / stride
    sample_period = stride / info.fps

    names = [
        name.strip()
        for name in args.names.split(",")
        if name.strip()
    ]

    client = InferenceHTTPClient(
        api_url=API_URL,
        api_key=api_key,
    ).configure(
        InferenceConfiguration(api_key_transport="header")
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    timeline_path = output_dir / f"{stem}_timeline.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    debug_path = output_dir / f"{stem}_annotated.mp4"
    temp_image = output_dir / ".temporary_inference_frame.jpg"

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    writer = cv2.VideoWriter(
        str(debug_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_sample_fps,
        (info.width, info.height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create {debug_path}")

    tracks = {}
    rows = []
    frame_number = 0
    sampled_number = 0
    next_track_id = 1

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            source_index = frame_number
            frame_number += 1

            if source_index % stride != 0:
                continue

            timestamp = source_index / info.fps
            sampled_number += 1

            candidates = []

            if cv2.imwrite(str(temp_image), frame):
                try:
                    response = client.infer(
                        str(temp_image),
                        model_id=MODEL_ID,
                    )
                    candidates = make_candidates(
                        response,
                        args.confidence,
                    )
                except Exception as error:
                    print(
                        f"Warning: inference failed at "
                        f"{timestamp:.2f}s: {error}"
                    )
            else:
                print(
                    f"Warning: could not save frame "
                    f"at {timestamp:.2f}s for inference."
                )

            matched, next_track_id = assign_tracks(
                candidates=candidates,
                tracks=tracks,
                frame_number=sampled_number,
                frame_width=info.width,
                frame_height=info.height,
                names=names,
                next_track_id=next_track_id,
            )

            for candidate in candidates:
                track_id = candidate["track_id"]
                track = tracks[track_id]

                track["bbox"] = candidate["bbox"]
                track["last_frame"] = sampled_number
                track["last_timestamp"] = timestamp

                row = {
                    "timestamp_seconds": round(timestamp, 3),
                    "second": int(timestamp),
                    "track_id": track_id,
                    "name": track["name"],
                    "raw_state": candidate["raw_state"],
                    "state": candidate["raw_state"],
                    "confidence": round(
                        candidate["confidence"], 4
                    ),
                    "detection_class": candidate["detection_class"],
                    "x1": round(candidate["bbox"][0], 1),
                    "y1": round(candidate["bbox"][1], 1),
                    "x2": round(candidate["bbox"][2], 1),
                    "y2": round(candidate["bbox"][3], 1),
                }
                rows.append(row)

                color = (
                    (0, 0, 255)
                    if candidate["raw_state"] == "SLEEPING"
                    else (0, 200, 0)
                )

                x1, y1, x2, y2 = map(
                    int, candidate["bbox"]
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
                    (
                        f"{track['name']} "
                        f"{candidate['raw_state']} "
                        f"{candidate['confidence']:.2f}"
                    ),
                    (max(0, x1), max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            # A briefly missing detection is unknown, not evidence of wakefulness.
            for track_id, track in tracks.items():
                if track_id in matched:
                    continue

                if sampled_number - track["last_frame"] <= 2:
                    rows.append(
                        {
                            "timestamp_seconds": round(timestamp, 3),
                            "second": int(timestamp),
                            "track_id": track_id,
                            "name": track["name"],
                            "raw_state": "UNCERTAIN",
                            "state": "UNCERTAIN",
                            "confidence": None,
                            "detection_class": "missing",
                            "x1": None,
                            "y1": None,
                            "x2": None,
                            "y2": None,
                        }
                    )

            cv2.putText(
                frame,
                f"Time: {timestamp:.2f}s",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            writer.write(frame)

    finally:
        capture.release()
        writer.release()
        temp_image.unlink(missing_ok=True)

    rows = smooth_states(rows, sample_period)

    columns = [
        "timestamp_seconds",
        "second",
        "track_id",
        "name",
        "raw_state",
        "state",
        "confidence",
        "detection_class",
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

    intervals = summarize_intervals(
        rows,
        info.duration_seconds,
        sample_period,
    )

    summary = {
        "input_video": str(video_path.resolve()),
        "model": MODEL_ID,
        "inference": "Roboflow hosted inference API",
        "duration_seconds": round(info.duration_seconds, 3),
        "sampling_fps": round(actual_sample_fps, 3),
        "sleep_intervals": intervals,
        "name_mapping": {
            track_id: track["name"]
            for track_id, track in tracks.items()
        },
        "notes": [
            (
                "Names are assigned by track creation order; when multiple "
                "people first appear together, detections are sorted "
                "left-to-right."
            ),
            (
                "AWAKE means the detector did not label that detection as "
                "sleeping; it is not an independent confirmation of wakefulness."
            ),
            (
                "The model and temporal smoothing have not been independently "
                "validated on a diverse test set. Results are visual estimates, "
                "not a physiological sleep diagnosis."
            ),
        ],
    }

    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print("\nPretrained sleep detector")
    print("-------------------------")
    print(f"Video: {video_path.resolve()}")
    print(f"Duration: {info.duration_seconds:.2f}s")
    print(f"Sampled frames: {sampled_number}")
    print(f"Tracks: {len(tracks)}")
    print(f"Sleep intervals: {len(intervals)}")

    for interval in intervals:
        print(
            f"  {interval['name']} "
            f"({interval['track_id']}): "
            f"{interval['start_seconds']:.2f}s - "
            f"{interval['end_seconds']:.2f}s"
        )

    print(f"\nTimeline: {timeline_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Annotated video: {debug_path.resolve()}")


if __name__ == "__main__":
    main()
