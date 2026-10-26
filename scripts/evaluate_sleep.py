import csv
import json
from collections import defaultdict
from pathlib import Path


ANNOTATIONS = Path("evaluation/annotations/intervals.csv")
OUTPUTS = Path("outputs")
RESULTS = Path("evaluation/results")


def load_annotations():
    intervals = defaultdict(list)

    with ANNOTATIONS.open("r", newline="") as file:
        reader = csv.DictReader(file)

        required = {
            "video",
            "start_seconds",
            "end_seconds",
            "reference_state",
            "notes",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Annotation file is missing columns: {sorted(missing)}"
            )

        for row in reader:
            start = float(row["start_seconds"])
            end = float(row["end_seconds"])
            state = row["reference_state"].strip().upper()

            if end <= start:
                raise ValueError(
                    f"Invalid interval for {row['video']}: {start}-{end}"
                )

            if state not in {"AWAKE", "SLEEPING", "UNCERTAIN"}:
                raise ValueError(
                    f"Invalid reference state: {state}"
                )

            intervals[row["video"]].append(
                {
                    "start": start,
                    "end": end,
                    "state": state,
                }
            )

    return intervals


def reference_at(timestamp, intervals):
    matches = [
        interval["state"]
        for interval in intervals
        if interval["start"] <= timestamp < interval["end"]
    ]

    if len(matches) > 1:
        raise ValueError(
            f"Overlapping annotations at {timestamp:.3f}s"
        )

    return matches[0] if matches else None


def calculate_metrics(rows):
    tp = fp = fn = tn = 0
    uncertain_count = 0
    exact_state_matches = 0
    evaluated = 0

    for truth, prediction in rows:
        if truth not in {"AWAKE", "SLEEPING"}:
            continue

        evaluated += 1

        if prediction == "UNCERTAIN":
            uncertain_count += 1

        predicted_sleep = prediction == "SLEEPING"
        actual_sleep = truth == "SLEEPING"

        if actual_sleep and predicted_sleep:
            tp += 1
        elif not actual_sleep and predicted_sleep:
            fp += 1
        elif actual_sleep and not predicted_sleep:
            fn += 1
        else:
            tn += 1

        if prediction == truth:
            exact_state_matches += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "evaluated_seconds": evaluated,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "binary_accuracy": round(
            (tp + tn) / evaluated, 4
        ) if evaluated else None,
        "exact_state_accuracy": round(
            exact_state_matches / evaluated, 4
        ) if evaluated else None,
        "uncertain_seconds": uncertain_count,
        "uncertain_rate": round(
            uncertain_count / evaluated, 4
        ) if evaluated else None,
    }


def main():
    if not ANNOTATIONS.is_file():
        raise FileNotFoundError(
            f"Annotations not found: {ANNOTATIONS}"
        )

    annotations = load_annotations()
    RESULTS.mkdir(parents=True, exist_ok=True)

    per_video = {}
    all_pairs = []

    for video, intervals in sorted(annotations.items()):
        timeline_path = (
            OUTPUTS / f"{video}_sleep_timeline.csv"
        )

        if not timeline_path.is_file():
            print(f"Skipping {video}: prediction timeline missing")
            continue

        pairs = []

        with timeline_path.open("r", newline="") as file:
            for row in csv.DictReader(file):
                start = float(row["start_seconds"])
                end = float(row["end_seconds"])

                # Use the midpoint of each one-second output bin.
                midpoint = (start + end) / 2.0
                truth = reference_at(midpoint, intervals)

                if truth is None or truth == "UNCERTAIN":
                    continue

                prediction = row["state"].strip().upper()
                pairs.append((truth, prediction))

        if not pairs:
            print(f"Skipping {video}: no annotated seconds to evaluate")
            continue

        per_video[video] = calculate_metrics(pairs)
        all_pairs.extend(pairs)

    if not per_video:
        raise ValueError(
            "No prediction timelines could be evaluated. "
            "Run scripts/classify_sleep.py first."
        )

    overall = calculate_metrics(all_pairs)

    columns = [
        "video",
        *next(iter(per_video.values())).keys(),
    ]

    metrics_path = RESULTS / "per_video_metrics.csv"

    with metrics_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for video, metrics in per_video.items():
            writer.writerow({"video": video, **metrics})

    overall_path = RESULTS / "overall_metrics.json"

    with overall_path.open("w") as file:
        json.dump(
            {
                "evaluation_method": (
                    "Per-second timeline midpoint matching. "
                    "UNCERTAIN predictions count as non-sleep for "
                    "binary metrics and are reported separately."
                ),
                "per_video": per_video,
                "overall": overall,
            },
            file,
            indent=2,
        )

    print("\nBaseline evaluation")
    print("-------------------")

    for video, metrics in per_video.items():
        print(
            f"{video:30s} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"F1={metrics['f1']:.3f} "
            f"FP={metrics['fp']:3d} "
            f"FN={metrics['fn']:3d} "
            f"uncertain={metrics['uncertain_rate']:.1%}"
        )

    print("\nOverall")
    for key, value in overall.items():
        print(f"{key}: {value}")

    print(f"\nPer-video metrics: {metrics_path.resolve()}")
    print(f"Overall metrics:   {overall_path.resolve()}")


if __name__ == "__main__":
    main()
