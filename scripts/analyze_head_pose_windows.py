import argparse
import csv
from pathlib import Path

import numpy as np


AXES = [
    "rotation_x_deg",
    "rotation_y_deg",
    "rotation_z_deg",
]


def read_rows(path):
    rows = []

    with path.open("r", newline="") as file:
        for row in csv.DictReader(file):
            try:
                timestamp = float(row["timestamp_seconds"])
            except (ValueError, TypeError):
                continue

            record = {"timestamp_seconds": timestamp}

            for axis in AXES:
                try:
                    record[axis] = float(row[axis])
                except (ValueError, TypeError):
                    record[axis] = np.nan

            rows.append(record)

    return sorted(rows, key=lambda row: row["timestamp_seconds"])


def median_or_none(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.median(values)) if values else None


def main():
    parser = argparse.ArgumentParser(
        description="Summarize head pose in time windows."
    )
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--baseline-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if args.window_seconds <= 0 or args.baseline_seconds <= 0:
        parser.error("Window and baseline durations must be positive.")

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*_head_pose.csv"))

    if not files:
        raise FileNotFoundError(
            f"No *_head_pose.csv files found in {input_dir.resolve()}"
        )

    summaries = []

    for path in files:
        rows = read_rows(path)

        if not rows:
            print(f"Skipping empty file: {path}")
            continue

        baseline_rows = [
            row for row in rows
            if row["timestamp_seconds"] < args.baseline_seconds
        ]

        baseline = {
            axis: median_or_none(
                [row[axis] for row in baseline_rows]
            )
            for axis in AXES
        }

        grouped = {}

        for row in rows:
            window_index = int(
                row["timestamp_seconds"] // args.window_seconds
            )
            grouped.setdefault(window_index, []).append(row)

        for window_index, window_rows in sorted(grouped.items()):
            start = window_index * args.window_seconds
            end = min(
                start + args.window_seconds,
                rows[-1]["timestamp_seconds"],
            )

            record = {
                "video": path.stem.removesuffix("_head_pose"),
                "window_start_seconds": round(start, 3),
                "window_end_seconds": round(end, 3),
                "sample_count": len(window_rows),
            }

            for axis in AXES:
                values = [
                    row[axis] for row in window_rows
                    if np.isfinite(row[axis])
                ]

                current_median = median_or_none(values)

                record[f"{axis}_median"] = current_median
                record[f"{axis}_q10"] = (
                    float(np.percentile(values, 10))
                    if values else None
                )
                record[f"{axis}_q90"] = (
                    float(np.percentile(values, 90))
                    if values else None
                )

                baseline_value = baseline[axis]

                record[f"{axis}_delta_from_baseline"] = (
                    current_median - baseline_value
                    if current_median is not None
                    and baseline_value is not None
                    else None
                )

            summaries.append(record)

    if not summaries:
        raise ValueError("No valid head-pose windows were produced.")

    output_path = input_dir / "head_pose_window_summary.csv"

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summaries)

    print("\nHead-pose window summary")
    print("------------------------")

    for row in summaries:
        print(
            f"{row['video']:30s} "
            f"{row['window_start_seconds']:5.1f}-"
            f"{row['window_end_seconds']:5.1f}s "
            f"n={row['sample_count']:3d}  "
            f"X={row['rotation_x_deg_median']}  "
            f"Y={row['rotation_y_deg_median']}  "
            f"Z={row['rotation_z_deg_median']}"
        )

    print(f"\nSaved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
