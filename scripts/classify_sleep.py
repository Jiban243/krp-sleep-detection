import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


def parse_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_features(csv_path):
    with csv_path.open("r", newline="") as file:
        reader = csv.DictReader(file)

        required = {
            "timestamp_seconds",
            "face_detected",
            "eye_a_ear",
            "eye_b_ear",
            "mean_ear",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Input CSV must contain: {sorted(required)}"
            )

        rows = []

        for row in reader:
            timestamp = parse_float(row["timestamp_seconds"])

            if timestamp is None:
                continue

            rows.append(
                {
                    "timestamp": timestamp,
                    "face_detected": parse_bool(row["face_detected"]),
                    "ear_a": parse_float(row["eye_a_ear"]),
                    "ear_b": parse_float(row["eye_b_ear"]),
                    "mean_ear": parse_float(row["mean_ear"]),
                }
            )

    rows.sort(key=lambda row: row["timestamp"])

    if not rows:
        raise ValueError("The feature CSV contains no usable rows.")

    return rows


def estimate_sample_period(rows):
    differences = [
        rows[i]["timestamp"] - rows[i - 1]["timestamp"]
        for i in range(1, len(rows))
        if rows[i]["timestamp"] > rows[i - 1]["timestamp"]
    ]

    return median(differences) if differences else 0.1


def classify_eye_observation(row, closed_threshold, open_threshold):
    if (
        not row["face_detected"]
        or row["ear_a"] is None
        or row["ear_b"] is None
    ):
        return "UNCERTAIN"

    a = row["ear_a"]
    b = row["ear_b"]

    if a < closed_threshold and b < closed_threshold:
        return "CLOSED"

    # One clearly open eye is evidence against sustained closure.
    if max(a, b) > open_threshold:
        return "OPEN"

    return "AMBIGUOUS"


def find_sleep_intervals(
    rows,
    labels,
    sample_period,
    duration,
    min_sleep_seconds,
    max_gap_seconds,
    min_closed_fraction,
):
    closed_indices = [
        i for i, label in enumerate(labels)
        if label == "CLOSED"
    ]

    if not closed_indices:
        return []

    groups = []
    current_group = [closed_indices[0]]

    for index in closed_indices[1:]:
        previous = current_group[-1]

        gap = rows[index]["timestamp"] - rows[previous]["timestamp"]

        if gap <= max_gap_seconds + 1e-9:
            current_group.append(index)
        else:
            groups.append(current_group)
            current_group = [index]

    groups.append(current_group)

    intervals = []

    for group in groups:
        start = rows[group[0]]["timestamp"]
        last_closed_time = rows[group[-1]]["timestamp"]
        end = min(duration, last_closed_time + sample_period)

        span = end - start

        if span < min_sleep_seconds:
            continue

        indices_in_span = [
            i for i, row in enumerate(rows)
            if start <= row["timestamp"] < end
        ]

        if not indices_in_span:
            continue

        closed_count = sum(
            labels[i] == "CLOSED"
            for i in indices_in_span
        )

        closed_fraction = closed_count / len(indices_in_span)

        if closed_fraction < min_closed_fraction:
            continue

        intervals.append(
            {
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(span, 3),
                "closed_fraction": round(closed_fraction, 4),
            }
        )

    return intervals


def main():
    parser = argparse.ArgumentParser(
        description="Detect sustained sleep-like eye closure."
    )

    parser.add_argument("--csv", required=True)

    parser.add_argument(
        "--closed-threshold",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--open-threshold",
        type=float,
        default=0.16,
    )

    parser.add_argument(
        "--min-sleep-seconds",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--min-closed-fraction",
        type=float,
        default=0.80,
    )

    args = parser.parse_args()

    if args.closed_threshold >= args.open_threshold:
        parser.error(
            "--closed-threshold must be lower than --open-threshold"
        )

    if args.min_sleep_seconds <= 0:
        parser.error("--min-sleep-seconds must be positive")

    if args.max_gap_seconds < 0:
        parser.error("--max-gap-seconds cannot be negative")

    if not 0 < args.min_closed_fraction <= 1:
        parser.error("--min-closed-fraction must be in (0, 1]")

    csv_path = Path(args.csv).expanduser()

    if not csv_path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    rows = load_features(csv_path)
    sample_period = estimate_sample_period(rows)

    duration = rows[-1]["timestamp"] + sample_period

    raw_labels = [
        classify_eye_observation(
            row,
            args.closed_threshold,
            args.open_threshold,
        )
        for row in rows
    ]

    intervals = find_sleep_intervals(
        rows=rows,
        labels=raw_labels,
        sample_period=sample_period,
        duration=duration,
        min_sleep_seconds=args.min_sleep_seconds,
        max_gap_seconds=args.max_gap_seconds,
        min_closed_fraction=args.min_closed_fraction,
    )

    # The recording is analysed offline. Once sustained closure is
    # confirmed, its detected interval is assigned retrospectively.
    frame_states = []

    for row, raw_label in zip(rows, raw_labels):
        timestamp = row["timestamp"]

        inside_sleep_interval = any(
            interval["start_seconds"] <= timestamp
            < interval["end_seconds"]
            for interval in intervals
        )

        if inside_sleep_interval:
            state = "SLEEPING"
        elif raw_label == "OPEN":
            state = "AWAKE"
        elif not row["face_detected"]:
            state = "UNCERTAIN"
        else:
            state = "UNCERTAIN"

        frame_states.append(state)

    # Aggregate frame observations into one-second judgements.
    timeline = []
    number_of_seconds = math.ceil(duration)

    for second in range(number_of_seconds):
        second_end = min(second + 1.0, duration)

        indices = [
            i for i, row in enumerate(rows)
            if second <= row["timestamp"] < second_end
        ]

        second_rows = [rows[i] for i in indices]
        second_labels = [raw_labels[i] for i in indices]
        second_states = [frame_states[i] for i in indices]

        if not second_rows:
            state = "UNCERTAIN"
            closed_fraction = None
            face_detection_rate = 0.0
            mean_ear = None
        else:
            count = len(second_rows)
            sleeping_count = second_states.count("SLEEPING")
            open_count = second_labels.count("OPEN")

            measurable_count = sum(
                label in {"CLOSED", "OPEN"}
                for label in second_labels
            )

            closed_count = second_labels.count("CLOSED")

            if sleeping_count >= 0.5 * count:
                state = "SLEEPING"
            elif open_count >= 0.5 * count:
                state = "AWAKE"
            else:
                state = "UNCERTAIN"

            closed_fraction = (
                round(closed_count / measurable_count, 4)
                if measurable_count
                else None
            )

            face_detection_rate = round(
                sum(row["face_detected"] for row in second_rows)
                / count,
                4,
            )

            ear_values = [
                row["mean_ear"]
                for row in second_rows
                if row["mean_ear"] is not None
            ]

            mean_ear = (
                round(sum(ear_values) / len(ear_values), 4)
                if ear_values
                else None
            )

        timeline.append(
            {
                "second": second,
                "start_seconds": round(second, 3),
                "end_seconds": round(second_end, 3),
                "state": state,
                "mean_ear": mean_ear,
                "closed_fraction": closed_fraction,
                "face_detection_rate": face_detection_rate,
                "sample_count": len(indices),
            }
        )

    output_stem = csv_path.stem.removesuffix("_eye_features")
    output_dir = csv_path.parent

    timeline_path = output_dir / f"{output_stem}_sleep_timeline.csv"
    summary_path = output_dir / f"{output_stem}_sleep_summary.json"

    timeline_columns = [
        "second",
        "start_seconds",
        "end_seconds",
        "state",
        "mean_ear",
        "closed_fraction",
        "face_detection_rate",
        "sample_count",
    ]

    with timeline_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=timeline_columns)
        writer.writeheader()
        writer.writerows(timeline)

    summary = {
        "input_feature_csv": str(csv_path.resolve()),
        "duration_seconds": round(duration, 3),
        "method": "EAR thresholding with sustained-closure filtering",
        "parameters": {
            "closed_threshold": args.closed_threshold,
            "open_threshold": args.open_threshold,
            "min_sleep_seconds": args.min_sleep_seconds,
            "max_gap_seconds": args.max_gap_seconds,
            "min_closed_fraction": args.min_closed_fraction,
        },
        "sleep_intervals": intervals,
        "interpretation": (
            "Visual sleep-like behaviour inferred from sustained eye "
            "closure; not a physiological sleep diagnosis."
        ),
    }

    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print("\nSleep detection baseline")
    print("------------------------")
    print(f"Input: {csv_path.resolve()}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sampled observations: {len(rows)}")
    print(f"Closed-eye threshold: {args.closed_threshold}")
    print(f"Minimum sustained closure: {args.min_sleep_seconds:.1f}s")
    print(f"Candidate sleep intervals: {len(intervals)}")

    if intervals:
        for interval in intervals:
            print(
                f"  {interval['start_seconds']:.2f}s - "
                f"{interval['end_seconds']:.2f}s "
                f"({interval['duration_seconds']:.2f}s; "
                f"closed fraction "
                f"{interval['closed_fraction']:.1%})"
            )
    else:
        print("  No sustained sleep-like intervals detected.")

    print(f"\nPer-second timeline: {timeline_path.resolve()}")
    print(f"Interval summary:    {summary_path.resolve()}")


if __name__ == "__main__":
    main()
