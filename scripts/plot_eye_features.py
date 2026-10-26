import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def read_features(csv_path):
    timestamps = []
    eye_a = []
    eye_b = []
    mean_ear = []

    with csv_path.open("r", newline="") as file:
        reader = csv.DictReader(file)

        required = {
            "timestamp_seconds",
            "eye_a_ear",
            "eye_b_ear",
            "mean_ear",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain these columns: {sorted(required)}"
            )

        for row in reader:
            try:
                timestamp = float(row["timestamp_seconds"])
            except (TypeError, ValueError):
                continue

            timestamps.append(timestamp)

            for column, values in (
                ("eye_a_ear", eye_a),
                ("eye_b_ear", eye_b),
                ("mean_ear", mean_ear),
            ):
                try:
                    value = float(row[column])
                    values.append(
                        value if math.isfinite(value) else float("nan")
                    )
                except (TypeError, ValueError):
                    values.append(float("nan"))

    return timestamps, eye_a, eye_b, mean_ear


def main():
    parser = argparse.ArgumentParser(
        description="Plot eye aspect ratio measurements over time."
    )
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()

    if not csv_path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    timestamps, eye_a, eye_b, mean_ear = read_features(csv_path)

    if not timestamps or not any(math.isfinite(v) for v in mean_ear):
        raise ValueError(
            "No valid EAR measurements found. Check the feature CSV."
        )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        timestamps,
        eye_a,
        marker=".",
        markersize=3,
        linewidth=1,
        label="Eye A",
    )

    ax.plot(
        timestamps,
        eye_b,
        marker=".",
        markersize=3,
        linewidth=1,
        label="Eye B",
    )

    ax.plot(
        timestamps,
        mean_ear,
        linewidth=1.5,
        label="Mean EAR",
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Eye Aspect Ratio")
    ax.set_title(f"Eye Opening Over Time — {csv_path.stem}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = csv_path.parent / (
        csv_path.stem.replace("_eye_features", "") + "_ear_plot.png"
    )

    fig.savefig(output_path, dpi=150)
    print(f"Plot saved: {output_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
