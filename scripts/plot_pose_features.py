import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FEATURES = [
    "nose_to_shoulder_dx_ratio",
    "nose_to_shoulder_dy_ratio",
    "torso_lean_degrees",
    "left_wrist_to_nose_ratio",
    "right_wrist_to_nose_ratio",
    "nose_motion_per_second",
    "shoulder_motion_per_second",
    "torso_motion_per_second",
    "pose_visibility",
]


def read_pose_csv(path):
    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)

        required = {
            "timestamp_seconds",
            "pose_detected",
            *FEATURES,
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing columns in {path}: {sorted(missing)}"
            )

        rows = []

        for row in reader:
            try:
                timestamp = float(row["timestamp_seconds"])
            except (ValueError, TypeError):
                continue

            record = {
                "timestamp_seconds": timestamp,
                "pose_detected": (
                    str(row["pose_detected"]).strip().lower()
                    in {"true", "1", "yes"}
                ),
            }

            for column in FEATURES:
                try:
                    value = float(row[column])
                    record[column] = (
                        value if math.isfinite(value) else np.nan
                    )
                except (ValueError, TypeError):
                    record[column] = np.nan

            rows.append(record)

    rows.sort(key=lambda row: row["timestamp_seconds"])

    if not rows:
        raise ValueError(f"No valid observations in {path}")

    return rows


def plot_feature(ax, rows, columns, title, ylabel):
    times = [row["timestamp_seconds"] for row in rows]

    for column, label in columns:
        values = [row[column] for row in rows]

        ax.plot(
            times,
            values,
            linewidth=1.2,
            label=label,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize pose and movement features."
    )
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    path = Path(args.csv).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Pose CSV not found: {path}")

    rows = read_pose_csv(path)

    times = [row["timestamp_seconds"] for row in rows]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 12),
        sharex=True,
    )

    plot_feature(
        axes[0],
        rows,
        [
            ("nose_to_shoulder_dx_ratio", "Nose horizontal offset"),
            ("nose_to_shoulder_dy_ratio", "Nose vertical offset"),
        ],
        "Head position relative to shoulders",
        "Normalized ratio",
    )

    plot_feature(
        axes[1],
        rows,
        [
            ("torso_lean_degrees", "Torso lean"),
        ],
        "Torso orientation",
        "Degrees",
    )

    plot_feature(
        axes[2],
        rows,
        [
            ("left_wrist_to_nose_ratio", "Left wrist to nose"),
            ("right_wrist_to_nose_ratio", "Right wrist to nose"),
        ],
        "Hand proximity to face",
        "Distance / shoulder width",
    )

    plot_feature(
        axes[3],
        rows,
        [
            ("nose_motion_per_second", "Nose motion"),
            ("shoulder_motion_per_second", "Shoulder motion"),
            ("torso_motion_per_second", "Torso motion"),
        ],
        "Movement over time",
        "Normalized movement / second",
    )

    axes[3].set_xlabel("Time (seconds)")

    fig.suptitle(path.stem, fontsize=14)
    fig.tight_layout()

    output_path = path.parent / (
        path.stem.replace("_pose_features", "")
        + "_pose_diagnostics.png"
    )

    fig.savefig(output_path, dpi=150)

    detected = sum(row["pose_detected"] for row in rows)

    print("\nPose feature summary")
    print("--------------------")
    print(f"Input: {path.resolve()}")
    print(f"Observations: {len(rows)}")
    print(f"Pose detection rate: {detected / len(rows):.1%}")

    for column in FEATURES:
        values = [
            row[column]
            for row in rows
            if math.isfinite(row[column])
        ]

        if values:
            print(
                f"{column}: "
                f"median={np.median(values):.4f}, "
                f"10th percentile={np.percentile(values, 10):.4f}, "
                f"90th percentile={np.percentile(values, 90):.4f}, "
                f"valid={len(values)}/{len(rows)}"
            )
        else:
            print(f"{column}: no valid observations")

    print(f"\nPlot saved: {output_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
