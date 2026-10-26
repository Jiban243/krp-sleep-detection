import csv
import json
from collections import defaultdict
from pathlib import Path

ANNOTATIONS = Path("evaluation/annotations/intervals.csv")
PREDICTIONS_DIR = Path("outputs/multi_person")
OUTPUT_JSON = Path("evaluation/multi_person_metrics.json")


def safe_divide(a, b):
    return a / b if b else None


def metrics(tp, fp, fn, tn):
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, tp + fp + fn + tn)

    f1 = (
        safe_divide(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )

    return {
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


def fmt(value):
    return "N/A" if value is None else f"{value:.3f}"


def main():
    if not ANNOTATIONS.exists():
        print("Annotations file not found:", ANNOTATIONS)
        return

    annotations = defaultdict(list)

    with ANNOTATIONS.open(newline="") as f:
        for row in csv.DictReader(f):
            annotations[row["video"]].append({
                "start": float(row["start_seconds"]),
                "end": float(row["end_seconds"]),
                "state": row["reference_state"].upper(),
            })

    per_video = {}
    overall = defaultdict(int)
    decided = defaultdict(int)
    missing = []

    for video, segments in annotations.items():
        path = PREDICTIONS_DIR / f"{video}_timeline.csv"

        if not path.exists():
            missing.append(video)
            continue

        counts = defaultdict(int)

        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                timestamp = float(row["timestamp_seconds"])
                matching = [
                    segment for segment in segments
                    if segment["start"] <= timestamp < segment["end"]
                ]

                if not matching:
                    continue

                actual = matching[0]["state"]
                predicted = row["state"].upper()

                if actual not in ("AWAKE", "SLEEPING"):
                    continue

                is_sleep_actual = actual == "SLEEPING"
                is_sleep_predicted = predicted == "SLEEPING"

                if is_sleep_actual and is_sleep_predicted:
                    counts["tp"] += 1
                elif not is_sleep_actual and is_sleep_predicted:
                    counts["fp"] += 1
                elif is_sleep_actual:
                    counts["fn"] += 1
                else:
                    counts["tn"] += 1

                counts["samples"] += 1
                counts["uncertain"] += predicted == "UNCERTAIN"

                if predicted != "UNCERTAIN":
                    counts["decided"] += 1
                    if is_sleep_actual and is_sleep_predicted:
                        counts["d_tp"] += 1
                    elif not is_sleep_actual and is_sleep_predicted:
                        counts["d_fp"] += 1
                    elif is_sleep_actual:
                        counts["d_fn"] += 1
                    else:
                        counts["d_tn"] += 1

        result = metrics(
            counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        )
        result.update({
            "samples": counts["samples"],
            "uncertain": counts["uncertain"],
            "coverage": safe_divide(
                counts["decided"], counts["samples"]
            ),
        })
        per_video[video] = result

        for key in ("tp", "fp", "fn", "tn", "samples", "uncertain", "decided"):
            overall[key] += counts[key]

        for key in ("d_tp", "d_fp", "d_fn", "d_tn"):
            decided[key] += counts[key]

    result = metrics(
        overall["tp"], overall["fp"], overall["fn"], overall["tn"]
    )
    result.update({
        "samples": overall["samples"],
        "uncertain": overall["uncertain"],
        "coverage": safe_divide(overall["decided"], overall["samples"]),
        "decided_only": metrics(
            decided["d_tp"], decided["d_fp"],
            decided["d_fn"], decided["d_tn"],
        ),
        "per_video": per_video,
        "missing_prediction_files": missing,
        "evaluation_note": (
            "Development clips only; thresholds were developed on this "
            "dataset. UNCERTAIN counts as not predicted sleeping in the "
            "overall confusion matrix. Decided-only metrics exclude "
            "UNCERTAIN samples."
        ),
    })

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2))

    print("\nPER-VIDEO RESULTS")
    print("=" * 88)
    print(
        f"{'Video':27} {'N':>4} {'TP':>4} {'FP':>4} "
        f"{'FN':>4} {'TN':>4} {'Coverage':>10} {'F1':>8}"
    )

    for video, row in per_video.items():
        print(
            f"{video:27} {row['samples']:4d} "
            f"{row['tp']:4d} {row['fp']:4d} "
            f"{row['fn']:4d} {row['tn']:4d} "
            f"{100 * row['coverage']:9.1f}% {fmt(row['f1']):>8}"
        )

    print("\nOVERALL (UNCERTAIN COUNTS AS NOT SLEEPING)")
    print(f"Samples:     {result['samples']}")
    print(
        f"TP={result['tp']}, FP={result['fp']}, "
        f"FN={result['fn']}, TN={result['tn']}"
    )
    for key in ("precision", "recall", "f1", "specificity", "accuracy"):
        print(f"{key.capitalize():13}{fmt(result[key])}")
    print(f"Coverage:    {100 * result['coverage']:.1f}%")

    d = result["decided_only"]
    print("\nDECIDED SAMPLES ONLY (UNCERTAIN EXCLUDED)")
    print(f"TP={d['tp']}, FP={d['fp']}, FN={d['fn']}, TN={d['tn']}")
    print(f"Precision: {fmt(d['precision'])}")
    print(f"Recall:    {fmt(d['recall'])}")
    print(f"F1:        {fmt(d['f1'])}")

    if missing:
        print("\nMissing prediction files:", ", ".join(missing))

    print("\nSaved metrics:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
