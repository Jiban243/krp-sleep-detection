import argparse
import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp

from src.video_reader import inspect_video


VISIBILITY_THRESHOLD = 0.35

LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (0, 11),
    (0, 12),
]

OUTPUT_COLUMNS = [
    "timestamp_seconds",
    "pose_detected",
    "pose_visibility",
    "nose_x_norm",
    "nose_y_norm",
    "shoulder_mid_x_norm",
    "shoulder_mid_y_norm",
    "hip_mid_x_norm",
    "hip_mid_y_norm",
    "shoulder_width_px",
    "nose_to_shoulder_dx_ratio",
    "nose_to_shoulder_dy_ratio",
    "torso_lean_degrees",
    "left_wrist_to_nose_ratio",
    "right_wrist_to_nose_ratio",
    "nose_motion_per_second",
    "shoulder_motion_per_second",
    "torso_motion_per_second",
]


def get_point(landmarks, index, width, height):
    """Return visible landmark coordinates in pixels and normalized units."""
    landmark = landmarks[index]
    visibility = float(
        getattr(landmark, "visibility", 1.0)
    )

    if (
        not math.isfinite(visibility)
        or visibility < VISIBILITY_THRESHOLD
    ):
        return None

    x_norm = float(landmark.x)
    y_norm = float(landmark.y)

    if not math.isfinite(x_norm) or not math.isfinite(y_norm):
        return None

    return {
        "x": x_norm * width,
        "y": y_norm * height,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "visibility": visibility,
    }


def midpoint(point_a, point_b):
    if point_a is None or point_b is None:
        return None

    return {
        "x": (point_a["x"] + point_b["x"]) / 2.0,
        "y": (point_a["y"] + point_b["y"]) / 2.0,
    }


def distance(point_a, point_b):
    if point_a is None or point_b is None:
        return None

    return math.hypot(
        point_a["x"] - point_b["x"],
        point_a["y"] - point_b["y"],
    )


def draw_pose(frame, landmarks):
    """Draw selected body landmarks and connections on a frame."""
    height, width = frame.shape[:2]
    points = {}

    for index in LANDMARKS.values():
        points[index] = get_point(
            landmarks, index, width, height
        )

    for start_index, end_index in CONNECTIONS:
        start = points.get(start_index)
        end = points.get(end_index)

        if start is not None and end is not None:
            cv2.line(
                frame,
                (
                    round(start["x"]),
                    round(start["y"]),
                ),
                (
                    round(end["x"]),
                    round(end["y"]),
                ),
                (0, 255, 0),
                2,
            )

    for point in points.values():
        if point is not None:
            cv2.circle(
                frame,
                (round(point["x"]), round(point["y"])),
                4,
                (0, 0, 255),
                -1,
            )


def extract_features(landmarks, width, height):
    """Compute posture and relative-position features for one frame."""
    points = {
        name: get_point(landmarks, index, width, height)
        for name, index in LANDMARKS.items()
    }

    nose = points["nose"]

    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]

    left_hip = points["left_hip"]
    right_hip = points["right_hip"]

    shoulder_mid = midpoint(
        left_shoulder, right_shoulder
    )

    hip_mid = midpoint(left_hip, right_hip)

    shoulder_width = distance(
        left_shoulder, right_shoulder
    )

    if shoulder_width is not None and shoulder_width < 1e-6:
        shoulder_width = None

    nose_dx_ratio = None
    nose_dy_ratio = None
    torso_lean_degrees = None

    if (
        nose is not None
        and shoulder_mid is not None
        and shoulder_width is not None
    ):
        nose_dx_ratio = (
            nose["x"] - shoulder_mid["x"]
        ) / shoulder_width

        nose_dy_ratio = (
            nose["y"] - shoulder_mid["y"]
        ) / shoulder_width

    if (
        shoulder_mid is not None
        and hip_mid is not None
    ):
        torso_lean_degrees = math.degrees(
            math.atan2(
                hip_mid["x"] - shoulder_mid["x"],
                hip_mid["y"] - shoulder_mid["y"],
            )
        )

    left_wrist_distance = distance(
        points["left_wrist"], nose
    )

    right_wrist_distance = distance(
        points["right_wrist"], nose
    )

    left_wrist_ratio = (
        left_wrist_distance / shoulder_width
        if left_wrist_distance is not None
        and shoulder_width is not None
        else None
    )

    right_wrist_ratio = (
        right_wrist_distance / shoulder_width
        if right_wrist_distance is not None
        and shoulder_width is not None
        else None
    )

    key_visibility = [
        points[name]["visibility"]
        for name in (
            "nose",
            "left_shoulder",
            "right_shoulder",
        )
        if points[name] is not None
    ]

    pose_visibility = (
        sum(key_visibility) / len(key_visibility)
        if key_visibility
        else None
    )

    return {
        "pose_detected": True,
        "pose_visibility": pose_visibility,
        "nose_x_norm": (
            nose["x_norm"] if nose else None
        ),
        "nose_y_norm": (
            nose["y_norm"] if nose else None
        ),
        "shoulder_mid_x_norm": (
            shoulder_mid["x"] / width
            if shoulder_mid else None
        ),
        "shoulder_mid_y_norm": (
            shoulder_mid["y"] / height
            if shoulder_mid else None
        ),
        "hip_mid_x_norm": (
            hip_mid["x"] / width
            if hip_mid else None
        ),
        "hip_mid_y_norm": (
            hip_mid["y"] / height
            if hip_mid else None
        ),
        "shoulder_width_px": shoulder_width,
        "nose_to_shoulder_dx_ratio": nose_dx_ratio,
        "nose_to_shoulder_dy_ratio": nose_dy_ratio,
        "torso_lean_degrees": torso_lean_degrees,
        "left_wrist_to_nose_ratio": left_wrist_ratio,
        "right_wrist_to_nose_ratio": right_wrist_ratio,
        "_nose_x_px": nose["x"] if nose else None,
        "_nose_y_px": nose["y"] if nose else None,
        "_shoulder_x_px": (
            shoulder_mid["x"] if shoulder_mid else None
        ),
        "_shoulder_y_px": (
            shoulder_mid["y"] if shoulder_mid else None
        ),
        "_torso_x_px": (
            (shoulder_mid["x"] + hip_mid["x"]) / 2.0
            if shoulder_mid is not None and hip_mid is not None
            else None
        ),
        "_torso_y_px": (
            (shoulder_mid["y"] + hip_mid["y"]) / 2.0
            if shoulder_mid is not None and hip_mid is not None
            else None
        ),
    }


def calculate_motion(previous, current, timestamp, previous_timestamp):
    """Estimate landmark movement normalized by shoulder width per second."""
    results = {
        "nose_motion_per_second": None,
        "shoulder_motion_per_second": None,
        "torso_motion_per_second": None,
    }

    if previous is None or previous_timestamp is None:
        return results

    dt = timestamp - previous_timestamp

    if dt <= 0:
        return results

    width = current["shoulder_width_px"]

    if width is None or width <= 1e-6:
        width = previous["shoulder_width_px"]

    if width is None or width <= 1e-6:
        return results

    pairs = [
        (
            "_nose_x_px",
            "_nose_y_px",
            "nose_motion_per_second",
        ),
        (
            "_shoulder_x_px",
            "_shoulder_y_px",
            "shoulder_motion_per_second",
        ),
        (
            "_torso_x_px",
            "_torso_y_px",
            "torso_motion_per_second",
        ),
    ]

    for x_key, y_key, output_key in pairs:
        values = [
            previous.get(x_key),
            previous.get(y_key),
            current.get(x_key),
            current.get(y_key),
        ]

        if any(value is None for value in values):
            continue

        displacement = math.hypot(
            current[x_key] - previous[x_key],
            current[y_key] - previous[y_key],
        )

        results[output_key] = displacement / width / dt

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract full-frame body posture and motion features."
        )
    )

    parser.add_argument("--video", required=True)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument(
        "--model",
        default="models/pose_landmarker_lite.task",
    )

    args = parser.parse_args()

    video_path = Path(args.video).expanduser()
    model_path = Path(args.model).expanduser()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Pose model not found: {model_path}"
        )

    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive")

    info = inspect_video(video_path)

    stride = max(1, round(info.fps / args.sample_fps))
    actual_sample_fps = info.fps / stride

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    csv_path = output_dir / f"{stem}_pose_features.csv"
    video_out_path = output_dir / f"{stem}_pose_debug.mp4"

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path.resolve())
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    writer = cv2.VideoWriter(
        str(video_out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_sample_fps,
        (info.width, info.height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create debug video: {video_out_path}"
        )

    rows = []
    frame_index = 0
    previous_features = None
    previous_timestamp = None

    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = capture.read()

                if not success:
                    break

                current_index = frame_index
                frame_index += 1

                if current_index % stride != 0:
                    continue

                timestamp = current_index / info.fps
                timestamp_ms = round(timestamp * 1000)

                height, width = frame.shape[:2]

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]

                    features = extract_features(
                        landmarks,
                        width,
                        height,
                    )

                    motion = calculate_motion(
                        previous_features,
                        features,
                        timestamp,
                        previous_timestamp,
                    )

                    features.update(motion)

                    draw_pose(frame, landmarks)

                    status_text = "POSE DETECTED"
                    status_color = (0, 200, 0)

                    previous_features = features.copy()
                    previous_timestamp = timestamp

                else:
                    features = {
                        column: None
                        for column in OUTPUT_COLUMNS
                    }

                    features["pose_detected"] = False
                    features["timestamp_seconds"] = round(
                        timestamp, 3
                    )

                    status_text = "NO POSE DETECTED"
                    status_color = (0, 0, 255)

                    # A missed detection breaks the motion sequence.
                    previous_features = None
                    previous_timestamp = None

                features["timestamp_seconds"] = round(
                    timestamp, 3
                )

                for y, text in (
                    (30, f"Time: {timestamp:.2f}s"),
                    (60, status_text),
                ):
                    cv2.putText(
                        frame,
                        text,
                        (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        status_color,
                        2,
                    )

                writer.write(frame)

                row = {
                    column: features.get(column)
                    for column in OUTPUT_COLUMNS
                }

                rows.append(row)

    finally:
        capture.release()
        writer.release()

    with csv_path.open("w", newline="") as file:
        writer_csv = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS,
        )
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    detected = sum(
        bool(row["pose_detected"])
        for row in rows
    )

    print("\nPose feature extraction")
    print("-----------------------")
    print(f"Input: {video_path.resolve()}")
    print(f"Duration: {info.duration_seconds:.2f}s")
    print(f"Sampling FPS: {actual_sample_fps:.3f}")
    print(f"Sampled frames: {len(rows)}")

    if rows:
        print(
            f"Pose detection rate: "
            f"{detected / len(rows):.1%}"
        )

    print(f"Feature CSV: {csv_path.resolve()}")
    print(f"Debug video: {video_out_path.resolve()}")
    print("No sleep classification has been performed.")


if __name__ == "__main__":
    main()
