import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


VALID_STATES = {"AWAKE", "SLEEPING", "UNCERTAIN"}


def load_annotations(path):
    annotations = defaultdict(list)

    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            video = row["video"].strip()
            start = float(row["start_seconds"])
            end = float(row["end_seconds"])
            state = row["reference_state"].strip().upper()

            if state not in VALID_STATES:
                raise ValueError(f"Invalid reference state: {state}")

            if end <= start:
                raise ValueError(
                    f"Invalid annotation interval for {video}"
                )

            annotations[video].append(
                {"start": start, "end": end, "state": state}
            )

    return annotations


def get_reference_state(timestamp, intervals):
    matches = [
        interval["state"]
        for interval in intervals
        if interval["start"] <= timestamp < interval["end"]
    ]

    if len(matches) > 1:
        raise ValueError(
            f"Overlapping reference annotations at {timestamp:.3f}s"
        )

    return matches[0] if matches else None


def calculate_binary_metrics(pairs):
    tp = fp = fn = tn = 0

    for truth, prediction in pairs:
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

    total = len(pairs)

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else 0.0 if precision == 0 and recall == 0 else None
    )

    accuracy = (tp + tn) / total if total else None

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
        "accuracy": accuracy,
    }


def round_metrics(metrics):
    return {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in metrics.items()
    }


def evaluate_video(video, intervals, predictions_dir):
    path = predictions_dir / f"{video}_sleep_timeline.csv"

    if not path.is_file():
        print(f"Skipping {video}: missing {path}")
        return None

    pairs = []

    with path.open("r", newline="") as file:
        for row in csv.DictReader(file):
            start = float(row["start_seconds"])
            end = float(row["end_seconds"])
            midpoint = (start + end) / 2.0

            truth = get_reference_state(midpoint, intervals)
            prediction = row["state"].strip().upper()

            if prediction not in VALID_STATES:
                raise ValueError(
                    f"Invalid prediction state: {prediction}"
                )

            if truth not in {"AWAKE", "SLEEPING"}:
                continue

            pairs.append((truth, prediction))

    if not pairs:
        print(f"Skipping {video}: no labelled seconds")
        return None

    decided = [
        (truth, prediction)
        for truth, prediction in pairs
        if prediction in {"AWAKE", "SLEEPING"}
    ]

    conservative = calculate_binary_metrics(pairs)
    selective = calculate_binary_metrics(decided)

    return {
        "annotated_seconds": len(pairs),
        "decided_seconds": len(decided),
        "coverage": len(decided) / len(pairs),
        "uncertain_seconds": len(pairs) - len(decided),
        "conservative": round_metrics(conservative),
        "decided_only": round_metrics(selective),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sleep timelines against reference annotations."
    )

    parser.add_argument(
        "--annotations",
        default="evaluation/annotations/intervals.csv",
    )
    parser.add_argument("--predictions-dir", default="outputs")
    parser.add_argument("--results-dir", default="evaluation/results")
    args = parser.parse_args()

    annotation_path = Path(args.annotations)
    predictions_dir = Path(args.predictions_dir)
    results_dir = Path(args.results_dir)

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_path}"
        )

    annotations = load_annotations(annotation_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    per_video = {}

    all_pairs = []
    all_decided = []

    for video, intervals in sorted(annotations.items()):
        result = evaluate_video(
            video,
            intervals,
            predictions_dir,
        )

        if result is None:
            continue

        per_video[video] = result

        timeline_path = (
            predictions_dir / f"{video}_sleep_timeline.csv"
        )

        with timeline_path.open("r", newline="") as file:
            for row in csv.DictReader(file):
                midpoint = (
                    float(row["start_seconds"])
                    + float(row["end_seconds"])
                ) / 2.0

                truth = get_reference_state(midpoint, intervals)
                prediction = row["state"].strip().upper()

                if truth not in {"AWAKE", "SLEEPING"}:
                    continue

                all_pairs.append((truth, prediction))

                if prediction in {"AWAKE", "SLEEPING"}:
                    all_decided.append((truth, prediction))

    if not per_video:
        raise ValueError(
            "No predictions were evaluated. Check filenames and directories."
        )

    uncertain_seconds = sum(
        result["uncertain_seconds"]
        for result in per_video.values()
    )

    annotated_seconds = sum(
        result["annotated_seconds"]
        for result in per_video.values()
    )

    summary = {
        "predictions_directory": str(predictions_dir.resolve()),
        "annotations": str(annotation_path.resolve()),
        "methodology": (
            "One-second timeline predictions are matched to the reference "
            "state at each bin midpoint. Conservative metrics treat UNCERTAIN "
            "as not predicting sleep. Decided-only metrics exclude UNCERTAIN "
            "predictions and must always be read alongside coverage."
        ),
        "overall": {
            "annotated_seconds": annotated_seconds,
            "uncertain_seconds": uncertain_seconds,
            "coverage": (
                (annotated_seconds - uncertain_seconds) / annotated_seconds
                if annotated_seconds else None
            ),
            "conservative": round_metrics(
                calculate_binary_metrics(all_pairs)
            ),
            "decided_only": round_metrics(
                calculate_binary_metrics(all_decided)
            ),
        },
        "per_video": per_video,
    }

    json_path = results_dir / "metrics.json"

    with json_path.open("w") as file:
        json.dump(summary, file, indent=2)

    csv_path = results_dir / "per_video_metrics.csv"

    columns = [
        "video",
        "annotated_seconds",
        "decided_seconds",
        "uncertain_seconds",
        "coverage",
        "conservative_precision",
        "conservative_recall",
        "conservative_f1",
        "conservative_specificity",
        "conservative_accuracy",
        "conservative_tp",
        "conservative_fp",
        "conservative_fn",
        "conservative_tn",
        "decided_only_precision",
        "decided_only_recall",
        "decided_only_f1",
        "decided_only_specificity",
        "decided_only_accuracy",
        "decided_only_tp",
        "decided_only_fp",
        "decided_only_fn",
        "decided_only_tn",
    ]

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for video, result in per_video.items():
            row = {
                "video": video,
                "annotated_seconds": result["annotated_seconds"],
                "decided_seconds": result["decided_seconds"],
                "uncertain_seconds": result["uncertain_seconds"],
                "coverage": result["coverage"],
            }

            for prefix, key in (
                ("conservative", "conservative"),
                ("decided_only", "decided_only"),
            ):
                metrics = result[key]

                for metric in (
                    "precision",
                    "recall",
                    "f1",
                    "specificity",
                    "accuracy",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                ):
                    if metric in metrics:
                        row[f"{prefix}_{metric}"] = metrics[metric]

            writer.writerow(row)

    print("\nSleep detection evaluation")
    print("--------------------------")

    for video, result in per_video.items():
        c = result["conservative"]
        d = result["decided_only"]

        print(
            f"{video:30s} "
            f"coverage={result['coverage']:.1%} "
            f"conservative_F1={c['f1']} "
            f"conservative_FP={c['fp']} "
            f"decided_only_F1={d['f1']}"
        )

    print("\nOverall")
    print(json.dumps(summary["overall"], indent=2))

    print(f"\nPer-video CSV: {csv_path.resolve()}")
    print(f"Metrics JSON:  {json_path.resolve()}")


if __name__ == "__main__":
    main()
