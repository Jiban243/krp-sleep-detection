import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


AXES = [
    "rotation_x_deg",
    "rotation_y_deg",
    "rotation_z_deg",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    path = Path(args.csv).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    times = []
    values = {axis: [] for axis in AXES}

    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                timestamp = float(row["timestamp_seconds"])
            except (ValueError, TypeError):
                continue

            times.append(timestamp)

            for axis in AXES:
                try:
                    value = float(row[axis])
                    if not math.isfinite(value):
                        value = float("nan")
                except (ValueError, TypeError):
                    value = float("nan")

                values[axis].append(value)

    if not times:
        raise ValueError("No timestamps were available to plot.")

    fig, ax = plt.subplots(figsize=(12, 5))

    for axis in AXES:
        ax.plot(
            times,
            values[axis],
            linewidth=1.2,
            label=axis.replace("_deg", ""),
        )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Rotation angle (degrees)")
    ax.set_title(f"Facial Rotation Over Time — {path.stem}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output = path.with_name(
        path.stem.replace("_head_pose", "")
        + "_head_pose_plot.png"
    )

    fig.savefig(output, dpi=150)
    print(f"Plot saved: {output.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
