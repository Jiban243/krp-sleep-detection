import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


COLUMNS = [
    "timestamp_seconds",
    "eye_a_ear",
    "eye_b_ear",
    "mean_ear",
    "eye_blink_left",
    "eye_blink_right",
    "eye_look_down_left",
    "eye_look_down_right",
]


def read_csv(csv_path):
    with csv_path.open("r", newline="") as file:
        reader = csv.DictReader(file)

        missing = set(COLUMNS) - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing columns in {csv_path}: {sorted(missing)}. "
                "Re-run extract_eye_features.py first."
            )

        rows = []

        for row in reader:
            try:
                timestamp = float(row["timestamp_seconds"])
            except (ValueError, TypeError):
                continue

            record = {"timestamp_seconds": timestamp}

            for column in COLUMNS:
                if column == "timestamp_seconds":
                    continue

                try:
                    value = float(row[column])
                    record[column] = (
                        value if math.isfinite(value) else float("nan")
                    )
                except (ValueError, TypeError):
                    record[column] = float("nan")

            rows.append(record)

    rows.sort(key=lambda row: row["timestamp_seconds"])

    if not rows:
        raise ValueError(f"No valid observations in {csv_path}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Plot EAR, blink and downward-gaze signals over time."
    )
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()

    if not csv_path.is_file():
        raise FileNotFoundError(f"File not found: {csv_path}")

    rows = read_csv(csv_path)

    times = [row["timestamp_seconds"] for row in rows]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
    )

    # Panel 1: eye opening.
    axes[0].plot(
        times,
        [row["eye_a_ear"] for row in rows],
        label="Eye A EAR",
        linewidth=1,
    )
    axes[0].plot(
        times,
        [row["eye_b_ear"] for row in rows],
        label="Eye B EAR",
        linewidth=1,
    )
    axes[0].plot(
        times,
        [row["mean_ear"] for row in rows],
        label="Mean EAR",
        linewidth=1.5,
    )
    axes[0].set_ylabel("EAR")
    axes[0].set_title("Eye opening")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: estimated eyelid-blink blendshapes.
    axes[1].plot(
        times,
        [row["eye_blink_left"] for row in rows],
        label="Left blink score",
        linewidth=1,
    )
    axes[1].plot(
        times,
        [row["eye_blink_right"] for row in rows],
        label="Right blink score",
        linewidth=1,
    )
    axes[1].set_ylabel("Blendshape score")
    axes[1].set_title("Estimated eyelid closure")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: estimated downward-gaze blendshapes.
    axes[2].plot(
        times,
        [row["eye_look_down_left"] for row in rows],
        label="Left downward-gaze score",
        linewidth=1,
    )
    axes[2].plot(
        times,
        [row["eye_look_down_right"] for row in rows],
        label="Right downward-gaze score",
        linewidth=1,
    )
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Blendshape score")
    axes[2].set_title("Estimated downward gaze")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(csv_path.stem, fontsize=14)
    fig.tight_layout()

    output_path = csv_path.parent / (
        csv_path.stem.replace("_eye_features", "")
        + "_temporal_diagnostics.png"
    )

    fig.savefig(output_path, dpi=150)
    print(f"Plot saved: {output_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
