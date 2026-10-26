import argparse
import csv
import json
from bisect import bisect_left
from pathlib import Path
from statistics import median

from src.video_reader import inspect_video

def read_csv(path):
    with path.open("r", newline="") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        try:
            row["timestamp_seconds"] = float(
                row["timestamp_seconds"]
            )
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid timestamp in {path}: "
                f"{row.get('timestamp_seconds')}"
            )

    return rows


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def get_sample_period(rows):
    times = sorted(
        float(row["timestamp_seconds"])
        for row in rows
    )

    differences = [
        times[i] - times[i - 1]
        for i in range(1, len(times))
        if times[i] > times[i - 1]
    ]

    return median(differences) if differences else 0.1


def nearest_pose(timestamp, pose_rows, pose_times, tolerance):
    index = bisect_left(pose_times, timestamp)
    candidates = []

    if index < len(pose_rows):
        candidates.append(pose_rows[index])

    if index > 0:
        candidates.append(pose_rows[index - 1])

    if not candidates:
        return None

    best = min(
        candidates,
        key=lambda row: abs(
            float(row["timestamp_seconds"]) - timestamp
        ),
    )

    if abs(float(best["timestamp_seconds"]) - timestamp) > tolerance:
        return None

    return best


def classify_observation(
    eye_row,
    pose_row,
    closed_threshold,
    open_threshold,
    x_min,
    x_max,
    z_min,
    z_max,
):
    if (
        not to_bool(eye_row.get("face_detected"))
        or to_float(eye_row.get("eye_a_ear")) is None
        or to_float(eye_row.get("eye_b_ear")) is None
    ):
        return "UNCERTAIN", False

    ear_a = float(eye_row["eye_a_ear"])
    ear_b = float(eye_row["eye_b_ear"])

    if ear_a < closed_threshold and ear_b < closed_threshold:
        if pose_row is None or not to_bool(
            pose_row.get("face_detected")
        ):
            return "CLOSED_UNSUPPORTED", False

        rotation_x = to_float(pose_row.get("rotation_x_deg"))
        rotation_z = to_float(pose_row.get("rotation_z_deg"))

        if rotation_x is None or rotation_z is None:
            return "CLOSED_UNSUPPORTED", False

        pose_compatible = (
            x_min <= rotation_x <= x_max
            and z_min <= rotation_z <= z_max
        )

        if pose_compatible:
            return "SUPPORTED_CLOSED", True

        return "CLOSED_UNSUPPORTED", False

    if max(ear_a, ear_b) > open_threshold:
        return "OPEN", False

    return "UNCERTAIN", False


def find_supported_intervals(
    rows,
    labels,
    sample_period,
    duration,
    min_duration,
    max_gap,
    min_fraction,
):
    closed_indices = [
        i for i, label in enumerate(labels)
        if label == "SUPPORTED_CLOSED"
    ]

    if not closed_indices:
        return []

    groups = []
    group = [closed_indices[0]]

    for index in closed_indices[1:]:
        previous = group[-1]
        gap = (
            rows[index]["timestamp_seconds"]
            - rows[previous]["timestamp_seconds"]
        )

        if gap <= max_gap + 1e-9:
            group.append(index)
        else:
            groups.append(group)
            group = [index]

    groups.append(group)
    intervals = []

    for group in groups:
        start = float(rows[group[0]]["timestamp_seconds"])
        last_time = float(rows[group[-1]]["timestamp_seconds"])
        end = min(duration, last_time + sample_period)

        if end - start < min_duration:
            continue

        in_span = [
            i for i, row in enumerate(rows)
            if start <= float(row["timestamp_seconds"]) < end
        ]

        if not in_span:
            continue

        support_fraction = sum(
            labels[i] == "SUPPORTED_CLOSED"
            for i in in_span
        ) / len(in_span)

        if support_fraction < min_fraction:
            continue

        intervals.append(
            {
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "supported_fraction": round(support_fraction, 4),
            }
        )

    return intervals


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Experimental sleep-like state detection using sustained "
            "eye closure and head orientation."
        )
    )

    parser.add_argument("--video", required=True)
    parser.add_argument("--eye-csv")
    parser.add_argument("--pose-csv")
    parser.add_argument("--output-dir", default="outputs/pose_aware")

    parser.add_argument("--closed-threshold", type=float, default=0.12)
    parser.add_argument("--open-threshold", type=float, default=0.16)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-gap", type=float, default=0.30)
    parser.add_argument("--min-fraction", type=float, default=0.80)

    # Exploratory ranges selected from the current recordings.
    parser.add_argument("--rotation-x-min", type=float, default=10.0)
    parser.add_argument("--rotation-x-max", type=float, default=23.0)
    parser.add_argument("--rotation-z-min", type=float, default=1.0)
    parser.add_argument("--rotation-z-max", type=float, default=10.0)

    args = parser.parse_args()

    if args.closed_threshold >= args.open_threshold:
        parser.error("Closed threshold must be below open threshold.")

    if args.min_duration <= 0 or args.max_gap < 0:
        parser.error("Minimum duration must be positive and max gap nonnegative.")

    if not 0 < args.min_fraction <= 1:
        parser.error("--min-fraction must be in (0, 1].")

    if args.rotation_x_min >= args.rotation_x_max:
        parser.error("Invalid rotation-X range.")

    if args.rotation_z_min >= args.rotation_z_max:
        parser.error("Invalid rotation-Z range.")

    video_path = Path(args.video).expanduser()
    stem = video_path.stem

    eye_path = (
        Path(args.eye_csv).expanduser()
        if args.eye_csv
        else Path("outputs") / f"{stem}_eye_features.csv"
    )

    pose_path = (
        Path(args.pose_csv).expanduser()
        if args.pose_csv
        else Path("outputs") / f"{stem}_head_pose.csv"
    )

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not eye_path.is_file():
        raise FileNotFoundError(f"Eye feature CSV not found: {eye_path}")

    if not pose_path.is_file():
        raise FileNotFoundError(f"Head-pose CSV not found: {pose_path}")

    eye_rows = read_csv(eye_path)
    pose_rows = read_csv(pose_path)

    if not eye_rows or not pose_rows:
        raise ValueError("Eye or head-pose CSV contains no observations.")

    eye_rows.sort(key=lambda row: float(row["timestamp_seconds"]))
    pose_rows.sort(key=lambda row: float(row["timestamp_seconds"]))

    duration = inspect_video(video_path).duration_seconds
    sample_period = get_sample_period(eye_rows)

    pose_times = [
        float(row["timestamp_seconds"])
        for row in pose_rows
    ]

    pose_period = get_sample_period(pose_rows)
    tolerance = max(0.075, 0.6 * min(sample_period, pose_period))

    labels = []
    pose_matched = []

    for eye_row in eye_rows:
        timestamp = float(eye_row["timestamp_seconds"])

        pose_row = nearest_pose(
            timestamp,
            pose_rows,
            pose_times,
            tolerance,
        )

        label, matched = classify_observation(
            eye_row=eye_row,
            pose_row=pose_row,
            closed_threshold=args.closed_threshold,
            open_threshold=args.open_threshold,
            x_min=args.rotation_x_min,
            x_max=args.rotation_x_max,
            z_min=args.rotation_z_min,
            z_max=args.rotation_z_max,
        )

        labels.append(label)
        pose_matched.append(pose_row if matched else None)

    intervals = find_supported_intervals(
        rows=eye_rows,
        labels=labels,
        sample_period=sample_period,
        duration=duration,
        min_duration=args.min_duration,
        max_gap=args.max_gap,
        min_fraction=args.min_fraction,
    )

    frame_states = []

    for eye_row, label in zip(eye_rows, labels):
        timestamp = float(eye_row["timestamp_seconds"])

        inside_interval = any(
            interval["start_seconds"] <= timestamp
            < interval["end_seconds"]
            for interval in intervals
        )

        if inside_interval:
            frame_states.append("SLEEPING")
        elif label == "OPEN":
            frame_states.append("AWAKE")
        else:
            frame_states.append("UNCERTAIN")

    timeline = []

    for second in range(int(duration) + (duration % 1 > 0)):
        start = float(second)
        end = min(start + 1.0, duration)

        indices = [
            i for i, row in enumerate(eye_rows)
            if start <= float(row["timestamp_seconds"]) < end
        ]

        if not indices:
            state = "UNCERTAIN"
            mean_ear = None
            closed_fraction = None
            pose_compatible_fraction = None
        else:
            states = [frame_states[i] for i in indices]
            raw_labels = [labels[i] for i in indices]

            sleeping_fraction = states.count("SLEEPING") / len(states)
            awake_fraction = states.count("AWAKE") / len(states)

            if sleeping_fraction >= 0.5:
                state = "SLEEPING"
            elif awake_fraction >= 0.5:
                state = "AWAKE"
            else:
                state = "UNCERTAIN"

            ears = [
                to_float(eye_rows[i].get("mean_ear"))
                for i in indices
            ]
            ears = [value for value in ears if value is not None]

            mean_ear = (
                round(sum(ears) / len(ears), 4)
                if ears else None
            )

            closed_fraction = round(
                sum(
                    raw_labels[j] in {
                        "SUPPORTED_CLOSED",
                        "CLOSED_UNSUPPORTED",
                    }
                    for j in range(len(raw_labels))
                ) / len(raw_labels),
                4,
            )

            pose_compatible_fraction = round(
                sum(
                    raw_labels[j] == "SUPPORTED_CLOSED"
                    for j in range(len(raw_labels))
                ) / len(raw_labels),
                4,
            )

        timeline.append(
            {
                "second": second,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "state": state,
                "mean_ear": mean_ear,
                "closed_eye_fraction": closed_fraction,
                "pose_supported_closed_fraction": pose_compatible_fraction,
                "sample_count": len(indices),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = output_dir / f"{stem}_sleep_timeline.csv"
    summary_path = output_dir / f"{stem}_sleep_summary.json"

    columns = list(timeline[0].keys()) if timeline else [
        "second", "start_seconds", "end_seconds", "state"
    ]

    with timeline_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(timeline)

    summary = {
        "video": str(video_path.resolve()),
        "duration_seconds": round(duration, 3),
        "method": "Experimental EAR + head-orientation gate + temporal persistence",
        "warning": (
            "Orientation ranges were chosen using the current development "
            "clips. This is not independent validation and may not generalize "
            "to other people, cameras, or sleeping postures."
        ),
        "parameters": {
            "closed_threshold": args.closed_threshold,
            "open_threshold": args.open_threshold,
            "min_duration_seconds": args.min_duration,
            "max_gap_seconds": args.max_gap,
            "min_supported_fraction": args.min_fraction,
            "rotation_x_range": [
                args.rotation_x_min,
                args.rotation_x_max,
            ],
            "rotation_z_range": [
                args.rotation_z_min,
                args.rotation_z_max,
            ],
        },
        "sleep_intervals": intervals,
        "interpretation": (
            "SLEEPING means sustained eye closure with head orientation "
            "compatible with the current staged-sleep example. UNCERTAIN "
            "means the available evidence does not satisfy the rule."
        ),
    }

    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print("\nPose-aware sleep detection")
    print("--------------------------")
    print(f"Video: {video_path.resolve()}")
    print(f"Sampled observations: {len(eye_rows)}")
    print(f"Pose match tolerance: {tolerance:.3f}s")
    print(f"Candidate sleep intervals: {len(intervals)}")

    for interval in intervals:
        print(
            f"  {interval['start_seconds']:.2f}s - "
            f"{interval['end_seconds']:.2f}s "
            f"({interval['duration_seconds']:.2f}s)"
        )

    print(
        "Uncertain seconds:",
        sum(row["state"] == "UNCERTAIN" for row in timeline),
        "/",
        len(timeline),
    )
    print(f"\nTimeline: {timeline_path.resolve()}")
    print(f"Summary:  {summary_path.resolve()}")


if __name__ == "__main__":
    main()
