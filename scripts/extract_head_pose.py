import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.video_reader import inspect_video


def crop_frame(frame):
    """Use the crop that worked for the existing face extractor."""
    height, width = frame.shape[:2]

    x0 = int(width * 0.05)
    x1 = int(width * 0.95)
    y0 = int(height * 0.15)
    y1 = int(height * 0.75)

    return frame[y0:y1, x0:x1]


def extract_rotation_angles(transform_matrix):
    """
    Extract approximate Euler rotation angles from a 4x4
    facial transformation matrix.

    These axes must be interpreted empirically for our camera setup.
    """
    matrix = np.asarray(transform_matrix, dtype=np.float64)

    if matrix.shape != (4, 4):
        raise ValueError(
            f"Expected a 4x4 transformation matrix, got {matrix.shape}"
        )

    rotation = matrix[:3, :3]

    # Project the matrix onto the nearest orthogonal rotation matrix.
    # This helps avoid treating small scale/shear components as rotation.
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt

    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    # OpenCV returns Euler angles in degrees.
    angles = cv2.RQDecomp3x3(rotation)[0]

    return tuple(float(angle) for angle in angles)


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3D facial rotation angles from a video."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument(
        "--model",
        default="models/face_landmarker.task",
    )
    args = parser.parse_args()

    video_path = Path(args.video).expanduser()
    model_path = Path(args.model).expanduser()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not model_path.is_file():
        raise FileNotFoundError(f"Face model not found: {model_path}")

    if args.sample_fps <= 0:
        parser.error("--sample-fps must be greater than zero")

    info = inspect_video(video_path)
    stride = max(1, round(info.fps / args.sample_fps))
    actual_sample_fps = info.fps / stride

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    csv_path = output_dir / f"{stem}_head_pose.csv"
    debug_path = output_dir / f"{stem}_head_pose_debug.mp4"

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path.resolve())
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )

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
        raise RuntimeError(f"Could not create debug video: {debug_path}")

    columns = [
        "timestamp_seconds",
        "face_detected",
        "rotation_x_deg",
        "rotation_y_deg",
        "rotation_z_deg",
    ]

    rows = []
    frame_index = 0

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(
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

                crop = crop_frame(frame)

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                result = landmarker.detect_for_video(
                    image,
                    timestamp_ms,
                )

                rotation_x = None
                rotation_y = None
                rotation_z = None

                detected = bool(result.face_landmarks)

                if (
                    detected
                    and result.facial_transformation_matrixes
                ):
                    (
                        rotation_x,
                        rotation_y,
                        rotation_z,
                    ) = extract_rotation_angles(
                        result.facial_transformation_matrixes[0]
                    )

                    label = (
                        f"Rotation X/Y/Z: "
                        f"{rotation_x:.1f}, "
                        f"{rotation_y:.1f}, "
                        f"{rotation_z:.1f} deg"
                    )
                    color = (0, 200, 0)
                else:
                    label = "HEAD POSE UNAVAILABLE"
                    color = (0, 0, 255)

                cv2.putText(
                    frame,
                    f"Time: {timestamp:.2f}s",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

                writer.write(frame)

                rows.append(
                    {
                        "timestamp_seconds": round(timestamp, 3),
                        "face_detected": detected,
                        "rotation_x_deg": rotation_x,
                        "rotation_y_deg": rotation_y,
                        "rotation_z_deg": rotation_z,
                    }
                )

    finally:
        capture.release()
        writer.release()

    with csv_path.open("w", newline="") as file:
        csv_writer = csv.DictWriter(file, fieldnames=columns)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    valid = [
        row for row in rows
        if row["rotation_x_deg"] is not None
    ]

    print("\nHead-pose extraction")
    print("--------------------")
    print(f"Input: {video_path.resolve()}")
    print(f"Duration: {info.duration_seconds:.2f}s")
    print(f"Sampled frames: {len(rows)}")

    if rows:
        print(
            "Valid rotation measurements: "
            f"{len(valid)}/{len(rows)} "
            f"({len(valid) / len(rows):.1%})"
        )

    for column in (
        "rotation_x_deg",
        "rotation_y_deg",
        "rotation_z_deg",
    ):
        values = [
            row[column]
            for row in rows
            if row[column] is not None
        ]

        if values:
            print(
                f"{column}: median={np.median(values):.2f}, "
                f"10th percentile={np.percentile(values, 10):.2f}, "
                f"90th percentile={np.percentile(values, 90):.2f}"
            )

    print(f"\nFeature CSV: {csv_path.resolve()}")
    print(f"Debug video: {debug_path.resolve()}")
    print("No sleep classification has been performed.")


if __name__ == "__main__":
    main()
