import argparse
from pathlib import Path

import cv2
import mediapipe as mp


EYE_INDICES = [
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--time", type=float, default=5.0)
    parser.add_argument(
        "--model",
        default="models/face_landmarker.task",
    )
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    video_path = Path(args.video).expanduser()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(f"Could not read frame at {args.time}s")

    h, w = frame.shape[:2]

    # Three views of the same frame, to test whether
    # the face's size relative to the image is the problem.
    views = {
        "full": frame,
        "upper": frame[:int(h * 0.80), :],
        "center_crop": frame[
            int(h * 0.15):int(h * 0.75),
            int(w * 0.05):int(w * 0.95),
        ],
    }

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for confidence in (0.50, 0.25):
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path.resolve())
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )

        print(f"\nConfidence threshold: {confidence}")

        with mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            for name, view in views.items():
                rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                result = landmarker.detect(image)
                annotated = view.copy()

                face_count = len(result.face_landmarks)
                print(f"  {name}: {face_count} face(s) detected")

                if face_count:
                    landmarks = result.face_landmarks[0]

                    for index in EYE_INDICES:
                        point = landmarks[index]
                        x = round(point.x * view.shape[1])
                        y = round(point.y * view.shape[0])

                        cv2.circle(
                            annotated,
                            (x, y),
                            3,
                            (0, 255, 0),
                            -1,
                        )

                    label = "FACE DETECTED"
                    color = (0, 200, 0)
                else:
                    label = "NO FACE DETECTED"
                    color = (0, 0, 255)

                cv2.putText(
                    annotated,
                    f"{name} | confidence={confidence}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                cv2.putText(
                    annotated,
                    label,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                out_path = output_dir / (
                    f"diagnostic_{video_path.stem}_"
                    f"{name}_c{confidence:.2f}.jpg"
                )
                cv2.imwrite(str(out_path), annotated)

    print("\nDiagnostic images saved in outputs/.")


if __name__ == "__main__":
    main()
