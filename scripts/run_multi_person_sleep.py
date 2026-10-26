import argparse
import csv
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def eye_aspect_ratio(landmarks, indices):
    """Compute eye aspect ratio from six Face Mesh landmarks."""
    points = np.array([
        [landmarks[i].x, landmarks[i].y] for i in indices
    ], dtype=float)

    horizontal = np.linalg.norm(points[0] - points[3])
    if horizontal < 1e-8:
        return None

    vertical = (
        np.linalg.norm(points[1] - points[5])
        + np.linalg.norm(points[2] - points[4])
    )
    return float(vertical / (2.0 * horizontal))


def facial_rotation(matrix):
    """Extract approximate Euler X and Z angles from a face matrix."""
    if matrix is None:
        return None, None

    array = np.asarray(matrix, dtype=float)
    if array.size < 16:
        return None, None

    rotation = array.reshape(4, 4)[:3, :3]

    rx = math.degrees(math.atan2(
        rotation[2, 1], rotation[2, 2]
    ))
    rz = math.degrees(math.atan2(
        rotation[1, 0], rotation[0, 0]
    ))

    return float(rx), float(rz)


def pose_observation(landmarks, width, height):
    """Create a bounding box, tracking center and nose for one pose."""
    visible = [
        lm for lm in landmarks
        if getattr(lm, "visibility", 1.0) is None
        or getattr(lm, "visibility", 1.0) > 0.15
    ]

    if not visible:
        visible = landmarks

    xs = [lm.x * width for lm in visible]
    ys = [lm.y * height for lm in visible]
    nose = landmarks[0]

    return {
        "center": (
            (min(xs) + max(xs)) / 2.0,
            (min(ys) + max(ys)) / 2.0,
        ),
        "nose": (nose.x * width, nose.y * height),
    }


def face_observations(result, width, height):
    """Extract EAR and head orientation for each detected face."""
    faces = []
    matrices = getattr(
        result, "facial_transformation_matrixes", []
    ) or []

    left_eye = [33, 160, 158, 133, 153, 144]
    right_eye = [362, 385, 387, 263, 373, 380]

    for face_index, landmarks in enumerate(result.face_landmarks):
        if len(landmarks) <= 387:
            continue

        left_ear = eye_aspect_ratio(landmarks, left_eye)
        right_ear = eye_aspect_ratio(landmarks, right_eye)

        valid_ears = [
            value for value in (left_ear, right_ear)
            if value is not None
        ]
        ear = float(np.mean(valid_ears)) if valid_ears else None

        matrix = (
            matrices[face_index]
            if face_index < len(matrices)
            else None
        )
        rx, rz = facial_rotation(matrix)

        # Mean face-landmark position is used for person association.
        center_x = float(np.mean([lm.x for lm in landmarks])) * width
        center_y = float(np.mean([lm.y for lm in landmarks])) * height

        faces.append({
            "center": (center_x, center_y),
            "ear": ear,
            "rotation_x": rx,
            "rotation_z": rz,
        })

    return faces


def associate_faces_with_poses(faces, poses, width, height):
    """Greedily match each face to its nearest pose nose."""
    pairs = []

    for face_index, face in enumerate(faces):
        fx, fy = face["center"]

        for pose_index, pose in enumerate(poses):
            px, py = pose["nose"]

            cost = (
                abs(fx - px) / width
                + 0.3 * abs(fy - py) / height
            )
            pairs.append((cost, face_index, pose_index))

    pairs.sort()
    matched_faces = set()
    matched_poses = set()
    result = {}

    for cost, face_index, pose_index in pairs:
        if cost > 0.18:
            continue
        if face_index in matched_faces or pose_index in matched_poses:
            continue

        result[pose_index] = faces[face_index]
        matched_faces.add(face_index)
        matched_poses.add(pose_index)

    return result


def assign_tracks(detections, tracks, names, width):
    """
    Maintain at most two tracks using nearest bounding-box centers.
    New identities are assigned left-to-right on the first detection.
    """
    assignments = {}

    if not tracks:
        for index in sorted(
            range(len(detections)),
            key=lambda i: detections[i]["center"][0],
        )[:2]:
            track_id = len(tracks)
            name = (
                names[track_id]
                if track_id < len(names)
                else f"Person {chr(65 + track_id)}"
            )
            tracks[track_id] = {
                "name": name,
                "center": detections[index]["center"],
                "missed": 0,
            }
            assignments[index] = track_id
        return assignments

    # Greedy minimum-distance matching between current detections and
    # previously observed tracks.
    pairs = []
    for detection_index, detection in enumerate(detections):
        for track_id, track in tracks.items():
            distance = math.dist(
                detection["center"], track["center"]
            )
            pairs.append((distance, detection_index, track_id))

    pairs.sort()
    used_detections = set()
    used_tracks = set()

    for distance, detection_index, track_id in pairs:
        if distance > width * 0.35:
            continue
        if detection_index in used_detections or track_id in used_tracks:
            continue

        assignments[detection_index] = track_id
        used_detections.add(detection_index)
        used_tracks.add(track_id)

    # Allocate a new identity if fewer than two identities exist.
    for index in sorted(
        range(len(detections)),
        key=lambda i: detections[i]["center"][0],
    ):
        if index in used_detections or len(tracks) >= 2:
            continue

        track_id = max(tracks.keys()) + 1
        name = (
            names[track_id]
            if track_id < len(names)
            else f"Person {chr(65 + track_id)}"
        )
        tracks[track_id] = {
            "name": name,
            "center": detections[index]["center"],
            "missed": 0,
        }
        assignments[index] = track_id
        used_detections.add(index)
        used_tracks.add(track_id)

    for detection_index, track_id in assignments.items():
        tracks[track_id]["center"] = detections[detection_index]["center"]
        tracks[track_id]["missed"] = 0

    for track_id in tracks:
        if track_id not in used_tracks:
            tracks[track_id]["missed"] += 1

    return assignments


def detect_face_for_pose(face_detector, pose_landmarks, frame):
    """Detect one face in an enlarged head crop guided by pose landmarks."""
    height, width = frame.shape[:2]
    nose = pose_landmarks[0]
    nose_x = int(nose.x * width)
    nose_y = int(nose.y * height)

    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]

    shoulder_span = math.hypot(
        (left_shoulder.x - right_shoulder.x) * width,
        (left_shoulder.y - right_shoulder.y) * height,
    )

    if shoulder_span < max(35, width * 0.035):
        shoulder_span = max(width * 0.12, 70)

    crop_width = max(int(shoulder_span * 0.9), int(width * 0.06), 64)
    crop_height = max(int(crop_width * 1.25), 80)

    x1 = max(0, nose_x - crop_width // 2)
    x2 = min(width, nose_x + crop_width // 2)
    y1 = max(0, nose_y - int(crop_height * 0.68))
    y2 = min(height, y1 + crop_height)

    if x2 - x1 < 32 or y2 - y1 < 32:
        return None

    crop = frame[y1:y2, x1:x2]
    enlarged = cv2.resize(
        crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
    )
    rgb = cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    result = face_detector.detect(image)
    detected_faces = face_observations(
        result, enlarged.shape[1], enlarged.shape[0]
    )

    if not detected_faces:
        return None

    center_x = enlarged.shape[1] / 2
    center_y = enlarged.shape[0] / 2

    return min(
        detected_faces,
        key=lambda face: math.hypot(
            face["center"][0] - center_x,
            face["center"][1] - center_y,
        ),
    )


def raw_state(ear, rx, rz, args):
    """Produce conservative per-frame evidence, not a final decision."""
    if ear is None:
        return "UNCERTAIN"

    if ear >= args.open_threshold:
        return "AWAKE"

    if ear <= args.closed_threshold and rx is not None and rz is not None:
        orientation_supported = (
            args.rotation_x_min <= rx <= args.rotation_x_max
            and args.rotation_z_min <= rz <= args.rotation_z_max
        )
        if orientation_supported:
            return "SLEEP_CANDIDATE"

    return "UNCERTAIN"


def smooth_person_rows(rows, min_duration):
    """Confirm only contiguous sleep-candidate runs long enough."""
    by_person = {}
    for row in rows:
        by_person.setdefault(row["track_id"], []).append(row)

    for person_rows in by_person.values():
        person_rows.sort(key=lambda row: row["timestamp_seconds"])

        for row in person_rows:
            row["state"] = (
                "AWAKE" if row["raw_state"] == "AWAKE"
                else "UNCERTAIN"
            )

        index = 0
        while index < len(person_rows):
            if person_rows[index]["raw_state"] != "SLEEP_CANDIDATE":
                index += 1
                continue

            end = index
            while (
                end + 1 < len(person_rows)
                and person_rows[end + 1]["raw_state"] == "SLEEP_CANDIDATE"
                and float(person_rows[end + 1]["timestamp_seconds"])
                    - float(person_rows[end]["timestamp_seconds"]) <= 1.1
            ):
                end += 1

            run_length = end - index + 1
            if run_length >= min_duration:
                for position in range(index, end + 1):
                    person_rows[position]["state"] = "SLEEPING"

            index = end + 1


def extract_intervals(rows, duration, sample_step):
    intervals = []
    by_track = {}

    for row in rows:
        by_track.setdefault(row["track_id"], []).append(row)

    for track_id, person_rows in by_track.items():
        person_rows.sort(key=lambda row: row["timestamp_seconds"])
        start = None
        last_timestamp = None
        person = person_rows[0]["person"]

        for row in person_rows:
            timestamp = float(row["timestamp_seconds"])

            if row["state"] == "SLEEPING":
                if (
                    start is None
                    or last_timestamp is None
                    or timestamp - last_timestamp > sample_step * 1.1
                ):
                    if start is not None:
                        intervals.append({
                            "person": person,
                            "track_id": track_id,
                            "start_seconds": round(start, 2),
                            "end_seconds": round(
                                min(last_timestamp + sample_step, duration), 2
                            ),
                            "duration_seconds": round(
                                min(last_timestamp + sample_step, duration)
                                - start, 2
                            ),
                        })
                    start = timestamp

                last_timestamp = timestamp

            elif start is not None:
                end_time = min(last_timestamp + sample_step, duration)
                intervals.append({
                    "person": person,
                    "track_id": track_id,
                    "start_seconds": round(start, 2),
                    "end_seconds": round(end_time, 2),
                    "duration_seconds": round(end_time - start, 2),
                })
                start = None
                last_timestamp = None

        if start is not None:
            end_time = min(last_timestamp + sample_step, duration)
            intervals.append({
                "person": person,
                "track_id": track_id,
                "start_seconds": round(start, 2),
                "end_seconds": round(end_time, 2),
                "duration_seconds": round(end_time - start, 2),
            })

    return intervals


def main():
    parser = argparse.ArgumentParser(
        description="Per-person experimental sleep-state detection"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="outputs/multi_person")
    parser.add_argument("--pose-model", default="models/pose_landmarker_lite.task")
    parser.add_argument("--face-model", default="models/face_landmarker.task")
    parser.add_argument("--names", default="Person A,Person B")
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--closed-threshold", type=float, default=0.12)
    parser.add_argument("--open-threshold", type=float, default=0.16)
    parser.add_argument("--min-duration", type=int, default=2)
    parser.add_argument("--rotation-x-min", type=float, default=10)
    parser.add_argument("--rotation-x-max", type=float, default=23)
    parser.add_argument("--rotation-z-min", type=float, default=1)
    parser.add_argument("--rotation-z-max", type=float, default=10)
    args = parser.parse_args()

    video_path = Path(args.video)
    pose_model = Path(args.pose_model)
    face_model = Path(args.face_model)
    output_dir = Path(args.output_dir)

    missing = [
        str(path) for path in (video_path, pose_model, face_model)
        if not path.exists()
    ]
    if missing:
        print("Missing required file(s):")
        for item in missing:
            print(" -", item)
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0

    if duration <= 0:
        print("Could not determine video duration.")
        cap.release()
        return

    names = [name.strip() for name in args.names.split(",") if name.strip()]
    if not names:
        names = ["Person A", "Person B"]

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(pose_model)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(face_model)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=2,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    tracks = {}
    all_rows = []
    sampled_frames = 0

    with (
        vision.PoseLandmarker.create_from_options(pose_options) as pose_detector,
        vision.FaceLandmarker.create_from_options(face_options) as face_detector,
    ):
        for second in range(0, math.ceil(duration), args.sample_step):
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = cap.read()
            if not ok:
                continue

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )
            timestamp_ms = second * 1000

            pose_result = pose_detector.detect_for_video(
                image, timestamp_ms
            )
            pose_landmark_sets = pose_result.pose_landmarks
            poses = [
                pose_observation(landmarks, width, height)
                for landmarks in pose_landmark_sets
            ]

            # Detect each face independently in its pose-guided head crop.
            face_by_pose = {}
            for pose_index, landmarks in enumerate(pose_landmark_sets):
                face = detect_face_for_pose(face_detector, landmarks, frame)
                if face is not None:
                    face_by_pose[pose_index] = face

            assignments = assign_tracks(
                poses, tracks, names, width
            )
            pose_by_track = {
                track_id: pose_index
                for pose_index, track_id in assignments.items()
            }
            sampled_frames += 1

            for track_id, track in sorted(tracks.items()):
                pose_index = pose_by_track.get(track_id)
                face = (
                    face_by_pose.get(pose_index)
                    if pose_index is not None
                    else None
                )

                ear = face["ear"] if face else None
                rx = face["rotation_x"] if face else None
                rz = face["rotation_z"] if face else None

                orientation_supported = (
                    rx is not None
                    and rz is not None
                    and args.rotation_x_min <= rx <= args.rotation_x_max
                    and args.rotation_z_min <= rz <= args.rotation_z_max
                )

                evidence = raw_state(ear, rx, rz, args)

                all_rows.append({
                    "timestamp_seconds": second,
                    "person": track["name"],
                    "track_id": track_id,
                    "pose_detected": pose_index is not None,
                    "face_detected": face is not None,
                    "ear": round(ear, 5) if ear is not None else "",
                    "rotation_x": round(rx, 3) if rx is not None else "",
                    "rotation_z": round(rz, 3) if rz is not None else "",
                    "head_orientation_support": (
                        orientation_supported if face else ""
                    ),
                    "raw_state": evidence,
                    "state": "UNCERTAIN",
                })

    cap.release()

    if not all_rows:
        print("No per-person rows were produced.")
        return

    smooth_person_rows(all_rows, args.min_duration)
    intervals = extract_intervals(
        all_rows, duration, args.sample_step
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    timeline_path = output_dir / f"{stem}_timeline.csv"
    intervals_path = output_dir / f"{stem}_intervals.json"
    summary_path = output_dir / f"{stem}_summary.json"

    fields = [
        "timestamp_seconds", "person", "track_id",
        "pose_detected", "face_detected", "ear",
        "rotation_x", "rotation_z", "head_orientation_support",
        "raw_state", "state",
    ]

    with timeline_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    with intervals_path.open("w") as file:
        json.dump(intervals, file, indent=2)

    summary = {
        "video": str(video_path),
        "duration_seconds": round(duration, 2),
        "sample_step_seconds": args.sample_step,
        "sampled_frames": sampled_frames,
        "pose_detections_total": sum(
            row["pose_detected"] for row in all_rows
        ),
        "face_detections_total": sum(
            row["face_detected"] for row in all_rows
        ),
        "people": {},
        "sleep_intervals": intervals,
        "method_note": (
            "Experimental EAR and head-orientation heuristic. "
            "Requires validation on independent real-world videos."
        ),
    }

    for track_id in sorted({row["track_id"] for row in all_rows}):
        person_rows = [
            row for row in all_rows if row["track_id"] == track_id
        ]
        summary["people"][person_rows[0]["person"]] = {
            "track_id": track_id,
            "pose_detected_seconds": sum(
                row["pose_detected"] for row in person_rows
            ),
            "face_detected_seconds": sum(
                row["face_detected"] for row in person_rows
            ),
            "state_counts": dict(Counter(
                row["state"] for row in person_rows
            )),
        }

    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print("\nMulti-person sleep analysis")
    print("---------------------------")
    print("Video:", video_path)
    print(f"Duration: {duration:.2f} seconds")
    print("Sampled frames:", sampled_frames)
    print("People tracked:", len(summary["people"]))
    print("Sleep intervals:", len(intervals))

    for person, info in summary["people"].items():
        print(f"\n{person} (track {info['track_id']})")
        print("  Pose detections:", info["pose_detected_seconds"])
        print("  Face detections:", info["face_detected_seconds"])
        print("  States:", info["state_counts"])

    for interval in intervals:
        print(
            f"  {interval['person']}: "
            f"{interval['start_seconds']:.1f}s–"
            f"{interval['end_seconds']:.1f}s "
            f"({interval['duration_seconds']:.1f}s)"
        )

    print("\nTimeline:", timeline_path)
    print("Intervals:", intervals_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
