from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


def inspect_video(video_path: str | Path) -> VideoInfo:
    path = Path(video_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    capture = cv2.VideoCapture(str(path))

    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            raise ValueError(f"Invalid video frame rate: {fps}")

        if frame_count <= 0:
            raise ValueError("Could not determine the video frame count.")

        return VideoInfo(
            path=str(path.resolve()),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=frame_count / fps,
        )
    finally:
        capture.release()


def sample_video(video_path: str | Path, sample_fps: float = 5.0):
    """Yield (timestamp_seconds, frame) at an approximate sampling rate."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero.")

    info = inspect_video(video_path)
    capture = cv2.VideoCapture(info.path)

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {info.path}")

    stride = max(1, round(info.fps / sample_fps))
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_index % stride == 0:
                timestamp = frame_index / info.fps
                yield timestamp, frame

            frame_index += 1
    finally:
        capture.release()
