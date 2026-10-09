from pathlib import Path
import csv
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

FILES = {
    "Baseline": RESULTS / "top5_results.csv",
    "Triplet": RESULTS / "triplet_top5_results.csv",
}
TARGET_CLASSES = {"cat", "bird", "dog", "airplane", "deer"}


def load_results(csv_path):
    data = defaultdict(list)

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["query_class"] not in TARGET_CLASSES:
                continue

            data[row["query_path"]].append({
                "query_class": row["query_class"],
                "rank": int(row["rank"]),
                "gallery_path": row["gallery_path"],
                "gallery_class": row["gallery_class"],
                "similarity": float(row["cosine_similarity"]),
            })

    for query_path in data:
        data[query_path].sort(key=lambda x: x["rank"])

    return data


def summarize(rows, true_class):
    top1_class = rows[0]["gallery_class"]
    correct_ranks = [
        r["rank"] for r in rows
        if r["gallery_class"] == true_class
    ]

    return {
        "top1_correct": top1_class == true_class,
        "top1_class": top1_class,
        "top5_hit": bool(correct_ranks),
        "first_correct_rank": min(correct_ranks) if correct_ranks else "",
    }


def main():
    baseline = load_results(FILES["Baseline"])
    triplet = load_results(FILES["Triplet"])

    common = sorted(set(baseline) & set(triplet))
    output = RESULTS / "cat_bird_error_comparison.csv"

    stats = {
        cls: {
            "total": 0,
            "baseline_top1": 0,
            "triplet_top1": 0,
            "baseline_top5": 0,
            "triplet_top5": 0,
            "improved_top1": 0,
            "regressed_top1": 0,
        }
        for cls in TARGET_CLASSES
    }

    improved = []
    regressed = []

    with open(output, "w", newline="") as f:
        fields = [
            "query_class", "query_path",
            "baseline_top1_class", "triplet_top1_class",
            "baseline_top1_correct", "triplet_top1_correct",
            "baseline_first_correct_rank", "triplet_first_correct_rank",
            "baseline_top5_hit", "triplet_top5_hit",
            "baseline_top5_classes", "triplet_top5_classes",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for path in common:
            cls = baseline[path][0]["query_class"]
            b = summarize(baseline[path], cls)
            t = summarize(triplet[path], cls)

            s = stats[cls]
            s["total"] += 1
            s["baseline_top1"] += int(b["top1_correct"])
            s["triplet_top1"] += int(t["top1_correct"])
            s["baseline_top5"] += int(b["top5_hit"])
            s["triplet_top5"] += int(t["top5_hit"])

            if not b["top1_correct"] and t["top1_correct"]:
                s["improved_top1"] += 1
                improved.append((cls, path, b, t))
            elif b["top1_correct"] and not t["top1_correct"]:
                s["regressed_top1"] += 1
                regressed.append((cls, path, b, t))

            writer.writerow({
                "query_class": cls,
                "query_path": path,
                "baseline_top1_class": b["top1_class"],
                "triplet_top1_class": t["top1_class"],
                "baseline_top1_correct": b["top1_correct"],
                "triplet_top1_correct": t["top1_correct"],
                "baseline_first_correct_rank": b["first_correct_rank"],
                "triplet_first_correct_rank": t["first_correct_rank"],
                "baseline_top5_hit": b["top5_hit"],
                "triplet_top5_hit": t["top5_hit"],
                "baseline_top5_classes": " | ".join(
                    r["gallery_class"] for r in baseline[path]
                ),
                "triplet_top5_classes": " | ".join(
                    r["gallery_class"] for r in triplet[path]
                ),
            })

    print("\n===== Cat / Bird Top-5 Error Analysis =====")
    for cls in ["airplane", "bird", "cat", "deer", "dog"]:
        s = stats[cls]
        n = s["total"]
        if not n:
            print(f"\n{cls}: no matching queries found")
            continue

        print(f"\n[{cls.upper()}] Query count: {n}")
        print(
            f"Top-1 correct: Baseline {s['baseline_top1']}/{n} "
            f"({100*s['baseline_top1']/n:.1f}%) -> "
            f"Triplet {s['triplet_top1']}/{n} "
            f"({100*s['triplet_top1']/n:.1f}%)"
        )
        print(
            f"Top-5 hit:    Baseline {s['baseline_top5']}/{n} "
            f"({100*s['baseline_top5']/n:.1f}%) -> "
            f"Triplet {s['triplet_top5']}/{n} "
            f"({100*s['triplet_top5']/n:.1f}%)"
        )
        print("Top-1 changed wrong -> correct:", s["improved_top1"])
        print("Top-1 changed correct -> wrong:", s["regressed_top1"])

    print("\nExamples where Top-1 improved (wrong -> correct):")
    for cls, path, b, t in improved[:8]:
        print(f"  {cls}: {Path(path).name}")
        print(
            f"    Baseline: {b['top1_class']} "
            f"(first correct rank={b['first_correct_rank']})"
        )
        print(
            f"    Triplet:  {t['top1_class']} "
            f"(first correct rank={t['first_correct_rank']})"
        )

    print("\nExamples where Top-1 regressed (correct -> wrong):")
    for cls, path, b, t in regressed[:8]:
        print(f"  {cls}: {Path(path).name}")
        print(
            f"    Baseline: {b['top1_class']} "
            f"(first correct rank={b['first_correct_rank']})"
        )
        print(
            f"    Triplet:  {t['top1_class']} "
            f"(first correct rank={t['first_correct_rank']})"
        )

    print("\nSaved detailed comparison:", output)


if __name__ == "__main__":
    main()
