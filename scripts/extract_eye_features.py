import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.video_reader import inspect_video


EYE_A = [33, 160, 158, 133, 153, 144]
EYE_B = [362, 385, 387, 263, 373, 380]

BLENDSHAPE_COLUMNS = [
    "eye_blink_left",
    "eye_blink_right",
    "eye_look_down_left",
    "eye_look_down_right",
]


def crop_frame(frame):
    """Crop the central region identified during face-detection testing."""
    height, width = frame.shape[:2]

    x0 = int(width * 0.05)
    x1 = int(width * 0.95)
    y0 = int(height * 0.15)
    y1 = int(height * 0.75)

    crop = frame[y0:y1, x0:x1]

    if crop.size == 0:
        raise ValueError("The selected face crop is empty.")

    return crop, x0, y0


def eye_aspect_ratio(landmarks, indices, width, height):
    """Calculate eye opening from six facial landmarks."""
    points = np.array(
        [
            [landmarks[i].x * width, landmarks[i].y * height]
            for i in indices
        ],
        dtype=np.float64,
    )

    p1, p2, p3, p4, p5, p6 = points

    horizontal = np.linalg.norm(p1 - p4)

    if horizontal < 1e-6:
        return None

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)

    return float(
        (vertical_1 + vertical_2) / (2.0 * horizontal)
    )


def draw_eye_contour(
    frame,
    landmarks,
    indices,
    crop_width,
    crop_height,
    offset_x,
    offset_y,
):
    """Draw crop-relative eye landmarks on the original video frame."""
    points = np.array(
        [
            [
                round(landmarks[i].x * crop_width) + offset_x,
                round(landmarks[i].y * crop_height) + offset_y,
            ]
            for i in indices
        ],
        dtype=np.int32,
    )

    cv2.polylines(
        frame,
        [points.reshape((-1, 1, 2))],
        isClosed=True,
        color=(0, 255, 255),
        thickness=2,
    )

    for x, y in points:
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)


def extract_blendshape_scores(result):
    """Extract blink and downward-gaze scores from MediaPipe output."""
    scores = {
        column: None
        for column in BLENDSHAPE_COLUMNS
    }

    if not result.face_blendshapes:
        return scores

    categories = result.face_blendshapes[0]

    category_scores = {
        category.category_name: float(category.score)
        for category in categories
    }

    name_mapping = {
        "eyeBlinkLeft": "eye_blink_left",
        "eyeBlinkRight": "eye_blink_right",
        "eyeLookDownLeft": "eye_look_down_left",
        "eyeLookDownRight": "eye_look_down_right",
    }

    for model_name, output_name in name_mapping.items():
        scores[output_name] = category_scores.get(model_name)

    return scores


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract eye geometry, blink scores, and downward-gaze "
            "scores from a video."
        )
    )

    parser.add_argument("--video", required=True)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument(
        "--model",
        default="models/face_landmarker.task",
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
            f"Model not found: {model_path}. Download it first."
        )

    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than zero.")

    info = inspect_video(video_path)

    stride = max(1, round(info.fps / args.sample_fps))
    actual_sample_fps = info.fps / stride

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem

    csv_path = output_dir / f"{stem}_eye_features.csv"
    video_out_path = output_dir / f"{stem}_eye_debug.mp4"

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path.resolve())
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
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

                crop, offset_x, offset_y = crop_frame(frame)
                crop_height, crop_width = crop.shape[:2]

                rgb = cv2.cvtColor(
                    crop,
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

                faces = result.face_landmarks

                ear_a = None
                ear_b = None
                mean_ear = None

                blendshapes = extract_blendshape_scores(result)

                if faces:
                    landmarks = faces[0]

                    ear_a = eye_aspect_ratio(
                        landmarks,
                        EYE_A,
                        crop_width,
                        crop_height,
                    )

                    ear_b = eye_aspect_ratio(
                        landmarks,
                        EYE_B,
                        crop_width,
                        crop_height,
                    )

                    valid_ears = [
                        value
                        for value in (ear_a, ear_b)
                        if value is not None
                    ]

                    if valid_ears:
                        mean_ear = float(np.mean(valid_ears))

                    draw_eye_contour(
                        frame,
                        landmarks,
                        EYE_A,
                        crop_width,
                        crop_height,
                        offset_x,
                        offset_y,
                    )

                    draw_eye_contour(
                        frame,
                        landmarks,
                        EYE_B,
                        crop_width,
                        crop_height,
                        offset_x,
                        offset_y,
                    )

                    status_text = "FACE DETECTED"
                    status_color = (0, 200, 0)

                    if ear_a is not None and ear_b is not None:
                        ear_text = (
                            f"EAR A: {ear_a:.3f} "
                            f"EAR B: {ear_b:.3f}"
                        )
                    else:
                        ear_text = "EAR unavailable"

                    blink_left = blendshapes["eye_blink_left"]
                    blink_right = blendshapes["eye_blink_right"]
                    down_left = blendshapes["eye_look_down_left"]
                    down_right = blendshapes["eye_look_down_right"]

                    if blink_left is not None and blink_right is not None:
                        blink_text = (
                            f"Blink L/R: {blink_left:.2f}/{blink_right:.2f}"
                        )
                    else:
                        blink_text = "Blink scores unavailable"

                    if down_left is not None and down_right is not None:
                        gaze_text = (
                            f"Look-down L/R: {down_left:.2f}/{down_right:.2f}"
                        )
                    else:
                        gaze_text = "Gaze scores unavailable"

                else:
                    status_text = "NO FACE DETECTED"
                    status_color = (0, 0, 255)
                    ear_text = "EAR unavailable"
                    blink_text = "Blink scores unavailable"
                    gaze_text = "Gaze scores unavailable"

                for y, text in (
                    (30, f"Time: {timestamp:.2f}s"),
                    (60, ear_text),
                    (90, blink_text),
                    (120, gaze_text),
                    (150, status_text),
                ):
                    cv2.putText(
                        frame,
                        text,
                        (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        status_color if y == 150 else (0, 255, 0),
                        2,
                    )

                writer.write(frame)

                rows.append(
                    {
                        "timestamp_seconds": round(timestamp, 3),
                        "face_detected": bool(faces),
                        "num_faces": len(faces),
                        "eye_a_ear": ear_a,
                        "eye_b_ear": ear_b,
                        "mean_ear": mean_ear,
                        **blendshapes,
                    }
                )

    finally:
        capture.release()
        writer.release()

    columns = [
        "timestamp_seconds",
        "face_detected",
        "num_faces",
        "eye_a_ear",
        "eye_b_ear",
        "mean_ear",
        *BLENDSHAPE_COLUMNS,
    ]

    with csv_path.open("w", newline="") as file:
        csv_writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    detected = [
        row for row in rows
        if row["face_detected"]
    ]

    valid_ears = [
        row["mean_ear"]
        for row in rows
        if row["mean_ear"] is not None
    ]

    print("\nVideo analysis")
    print("--------------")
    print(f"Input: {video_path.resolve()}")
    print(f"Duration: {info.duration_seconds:.2f} seconds")
    print(f"Resolution: {info.width} x {info.height}")
    print(f"Original FPS: {info.fps:.3f}")
    print(f"Sampling FPS: {actual_sample_fps:.3f}")
    print(f"Sampled frames: {len(rows)}")

    if rows:
        detection_rate = 100 * len(detected) / len(rows)
        print(f"Face detection rate: {detection_rate:.1f}%")
    else:
        print("Face detection rate: unavailable")

    if valid_ears:
        print(f"Valid EAR measurements: {len(valid_ears)}")
        print(f"Mean EAR: {np.mean(valid_ears):.4f}")
        print(f"Median EAR: {np.median(valid_ears):.4f}")
        print(
            "10th percentile EAR: "
            f"{np.percentile(valid_ears, 10):.4f}"
        )
        print(
            "90th percentile EAR: "
            f"{np.percentile(valid_ears, 90):.4f}"
        )
    else:
        print("No usable eye measurements were extracted.")

    for column in BLENDSHAPE_COLUMNS:
        values = [
            row[column]
            for row in rows
            if row[column] is not None
        ]

        if values:
            print(
                f"{column}: mean={np.mean(values):.4f}, "
                f"min={np.min(values):.4f}, "
                f"max={np.max(values):.4f}"
            )
        else:
            print(f"{column}: no valid measurements")

    print(f"\nFeature CSV: {csv_path.resolve()}")
    print(f"Debug video: {video_out_path.resolve()}")
    print("No sleep classification has been performed.")


if __name__ == "__main__":
    main()
