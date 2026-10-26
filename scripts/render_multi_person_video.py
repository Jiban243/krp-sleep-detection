import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from run_multi_person_sleep import assign_tracks, pose_observation


COLORS = {
    "AWAKE": (60, 200, 60),
    "SLEEPING": (50, 50, 230),
    "UNCERTAIN": (0, 190, 255),
}


def get_box(landmarks, width, height):
    points = []
    for lm in landmarks:
        visibility = getattr(lm, "visibility", 1.0)
        if visibility is None or visibility > 0.15:
            x = max(0, min(width - 1, int(lm.x * width)))
            y = max(0, min(height - 1, int(lm.y * height)))
            points.append((x, y))

    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    padding = 12

    return (
        max(0, min(xs) - padding),
        max(0, min(ys) - padding),
        min(width - 1, max(xs) + padding),
        min(height - 1, max(ys) + padding),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render an annotated sleep-state video"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--timeline", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--pose-model",
        default="models/pose_landmarker_lite.task",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    timeline_path = (
        Path(args.timeline)
        if args.timeline
        else Path("outputs/multi_person") /
        f"{video_path.stem}_timeline.csv"
    )
    output_path = (
        Path(args.output)
        if args.output
        else Path("outputs/multi_person") /
        f"{video_path.stem}_annotated.mp4"
    )
    model_path = Path(args.pose_model)

    for path in (video_path, timeline_path, model_path):
        if not path.exists():
            print("Required file not found:", path)
            return

    with timeline_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print("Timeline CSV contains no rows:", timeline_path)
        return

    state_history = {}
    names_by_track = {}

    for row in rows:
        track_id = int(row["track_id"])
        second = int(float(row["timestamp_seconds"]))
        state_history.setdefault(track_id, {})[second] = row["state"]
        names_by_track[track_id] = row["person"]

    track_ids = sorted(names_by_track)
    names = [
        names_by_track.get(track_id, f"Person {track_id + 1}")
        for track_id in range(max(track_ids) + 1)
    ]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print("Could not initialize output video writer:", output_path)
        cap.release()
        return

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    tracks = {}
    frame_index = 0

    with vision.PoseLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            elapsed = frame_index / fps
            second = int(elapsed)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )
            result = detector.detect_for_video(
                image, int(round(elapsed * 1000))
            )

            landmarks_list = result.pose_landmarks
            detections = [
                pose_observation(landmarks, width, height)
                for landmarks in landmarks_list
            ]
            assignments = assign_tracks(
                detections, tracks, names, width
            )

            for detection_index, track_id in assignments.items():
                landmarks = landmarks_list[detection_index]
                box = get_box(landmarks, width, height)
                if box is None:
                    continue

                history = state_history.get(track_id, {})
                eligible = [
                    timestamp for timestamp in history
                    if timestamp <= second
                ]
                state = (
                    history[max(eligible)]
                    if eligible
                    else "UNCERTAIN"
                )

                person = tracks[track_id]["name"]
                x1, y1, x2, y2 = box
                color = COLORS.get(state, COLORS["UNCERTAIN"])

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), color, 2
                )

                label = f"{person}: {state}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_width, text_height), baseline = (
                    cv2.getTextSize(label, font, 0.65, 2)
                )

                label_y = max(text_height + 8, y1)
                cv2.rectangle(
                    frame,
                    (x1, label_y - text_height - 8),
                    (min(width - 1, x1 + text_width + 10), label_y + baseline),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, label_y - 4),
                    font,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"Time: {elapsed:.1f}s",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frame_index += 1

    cap.release()
    writer.release()

    print("Annotated video:", output_path)
    print(f"Frames rendered: {frame_index}/{frame_count}")
    print(f"Output FPS: {fps:.2f}")
    print("Timeline used:", timeline_path)
    print(
        "Note: labels use one-second predictions held between samples. "
        "Identity switches may occur if people cross or overlap."
    )


if __name__ == "__main__":
    main()
