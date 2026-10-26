import csv
import json
from pathlib import Path


ANNOTATIONS = Path("evaluation/annotations/intervals.csv")
PREDICTIONS = Path("outputs/pretrained")
RESULTS = Path("evaluation/results_pretrained")


def load_annotations():
    result = {}

    with ANNOTATIONS.open("r", newline="") as file:
        for row in csv.DictReader(file):
            video = row["video"].strip()

            result.setdefault(video, []).append(
                {
                    "start": float(row["start_seconds"]),
                    "end": float(row["end_seconds"]),
                    "state": row["reference_state"].strip().upper(),
                }
            )

    return result


def reference_at(timestamp, intervals):
    for item in intervals:
        if item["start"] <= timestamp < item["end"]:
            return item["state"]

    return None


def metrics(pairs):
    tp = fp = fn = tn = 0
    uncertain = 0

    for truth, prediction in pairs:
        if prediction == "UNCERTAIN":
            uncertain += 1

        actual_sleep = truth == "SLEEPING"
        predicted_sleep = prediction == "SLEEPING"

        if actual_sleep and predicted_sleep:
            tp += 1
        elif not actual_sleep and predicted_sleep:
            fp += 1
        elif actual_sleep and not predicted_sleep:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None

    if precision is not None and recall is not None:
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    else:
        f1 = None

    total = len(pairs)
    decided = total - uncertain

    return {
        "seconds": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "conservative_accuracy": (
            (tp + tn) / total if total else None
        ),
        "uncertain_seconds": uncertain,
        "coverage": decided / total if total else None,
    }


def main():
    annotations = load_annotations()
    RESULTS.mkdir(parents=True, exist_ok=True)

    per_video = {}
    overall_pairs = []

    for video, intervals in sorted(annotations.items()):
        timeline_path = PREDICTIONS / f"{video}_timeline.csv"
        summary_path = PREDICTIONS / f"{video}_summary.json"

        if not timeline_path.is_file() or not summary_path.is_file():
            print(f"Skipping {video}: pretrained output missing")
            continue

        with summary_path.open() as file:
            summary = json.load(file)

        period = 1.0 / float(summary["sampling_fps"])
        duration = max(item["end"] for item in intervals)

        rows = []

        with timeline_path.open("r", newline="") as file:
            rows = list(csv.DictReader(file))

        # The current annotation file is clip-level and suitable for
        # these single-person recordings. Multi-person evaluation needs
        # identity-specific reference annotations.
        track_ids = {row["track_id"] for row in rows}

        if len(track_ids) > 1:
            print(
                f"Skipping {video}: multiple tracks require "
                "person-specific ground-truth annotations"
            )
            continue

        # Reconstruct the sampling grid so missing detections count
        # as UNCERTAIN rather than silently disappearing from evaluation.
        pairs = []
        count = int(duration / period) + 1

        for index in range(count):
            timestamp = index * period

            if timestamp >= duration:
                continue

            truth = reference_at(timestamp, intervals)

            if truth not in {"AWAKE", "SLEEPING"}:
                continue

            if rows:
                nearest = min(
                    rows,
                    key=lambda row: abs(
                        float(row["timestamp_seconds"]) - timestamp
                    ),
                )

                if abs(
                    float(nearest["timestamp_seconds"]) - timestamp
                ) <= period * 0.55:
                    prediction = nearest["state"].strip().upper()
                else:
                    prediction = "UNCERTAIN"
            else:
                prediction = "UNCERTAIN"

            if prediction not in {"AWAKE", "SLEEPING", "UNCERTAIN"}:
                prediction = "UNCERTAIN"

            pairs.append((truth, prediction))

        if not pairs:
            print(f"Skipping {video}: no labelled samples")
            continue

        per_video[video] = metrics(pairs)
        overall_pairs.extend(pairs)

    if not per_video:
        raise ValueError(
            "No predictions were evaluated. Run the pretrained model first."
        )

    output = {
        "evaluation_type": (
            "Held-out status not claimed: these are development recordings "
            "from the same person used during pipeline development."
        ),
        "per_video": per_video,
        "overall": metrics(overall_pairs),
    }

    json_path = RESULTS / "metrics.json"

    with json_path.open("w") as file:
        json.dump(output, file, indent=2)

    print("\nPretrained detector evaluation")
    print("------------------------------")

    for video, item in per_video.items():
        print(
            f"{video:30s} "
            f"precision={item['precision']} "
            f"recall={item['recall']} "
            f"F1={item['f1']} "
            f"FP={item['fp']} "
            f"FN={item['fn']} "
            f"coverage={item['coverage']:.1%}"
        )

    print("\nOverall")
    print(json.dumps(output["overall"], indent=2))
    print(f"\nSaved metrics: {json_path.resolve()}")


if __name__ == "__main__":
    main()
