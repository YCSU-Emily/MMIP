import csv
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("week10/assignment3_evaluation/videorag")

INPUT = BASE / "videorag_evaluation_results.json"
OUTPUT_JSON = BASE / "weighted_rrf_results.json"
OUTPUT_CSV = BASE / "weighted_rrf_summary.csv"
OUTPUT_MD = BASE / "weighted_rrf_report.md"

RRF_CONSTANT = 60
TOP_K = 8

EXPERIMENTS = {
    "baseline_1.0_1.0": {"speech": 1.0, "ocr": 1.0},
    "weighted_A_1.5_1.0": {"speech": 1.5, "ocr": 1.0},
    "weighted_B_2.0_1.0": {"speech": 2.0, "ocr": 1.0},
    "weighted_C_1.0_1.5": {"speech": 1.0, "ocr": 1.5},
}


def load_data():
    with INPUT.open("r", encoding="utf-8") as f:
        return json.load(f)


def fuse_candidates(question, speech_weight, ocr_weight):
    """Fuse the saved Speech/OCR candidate lists without model inference."""
    candidates = {}

    sources = [
        ("speech_only", speech_weight),
        ("ocr_only", ocr_weight),
    ]

    for source_name, weight in sources:
        for item in question[source_name]:
            evidence_id = item["id"]
            rank = item["rank"]

            if evidence_id not in candidates:
                candidates[evidence_id] = {
                    "id": evidence_id,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "distance": item.get("distance", float("inf")),
                    "speech_rank": None,
                    "ocr_rank": None,
                    "rrf_score": 0.0,
                }

            modality_rank_key = (
                "speech_rank"
                if source_name == "speech_only"
                else "ocr_rank"
            )

            candidates[evidence_id][modality_rank_key] = rank
            candidates[evidence_id]["rrf_score"] += (
                weight / (RRF_CONSTANT + rank)
            )

    ranked = sorted(
        candidates.values(),
        key=lambda x: (
            -x["rrf_score"],
            x["distance"],
            x["id"],
        ),
    )

    return ranked[:TOP_K]


def get_gold(question):
    return (
        set(question["gold_speech_ids"])
        | set(question["gold_ocr_ids"])
    )


def calculate_metrics(details, method_name, question_ids=None):
    selected = [
        q for q in details
        if question_ids is None or q["question_id"] in question_ids
    ]

    recall_counts = {1: 0, 3: 0, 5: 0}
    hit_counts = {1: 0, 3: 0, 5: 0}
    total_gold = 0
    reciprocal_rank_sum = 0

    for q in selected:
        gold = get_gold(q)
        retrieved = q["methods"][method_name]
        retrieved_ids = [item["id"] for item in retrieved]

        total_gold += len(gold)

        for k in (1, 3, 5):
            found = gold.intersection(retrieved_ids[:k])
            recall_counts[k] += len(found)

            if found:
                hit_counts[k] += 1

        first_rank = next(
            (
                rank
                for rank, item in enumerate(retrieved, 1)
                if item["id"] in gold
            ),
            None,
        )

        if first_rank is not None:
            reciprocal_rank_sum += 1.0 / first_rank

    if not selected or total_gold == 0:
        return {}

    metrics = {
        "eligible_questions": len(selected),
        "total_gold_evidence": total_gold,
    }

    for k in (1, 3, 5):
        metrics[f"recall@{k}"] = recall_counts[k] / total_gold
        metrics[f"hitrate@{k}"] = hit_counts[k] / len(selected)

    metrics["mrr"] = reciprocal_rank_sum / len(selected)
    return metrics


def main():
    data = load_data()
    source_details = data["details"]

    print("=" * 72)
    print("WEIGHTED RRF EXPERIMENT")
    print("=" * 72)
    print("Questions:", len(source_details))
    print("RRF constant:", RRF_CONSTANT)
    print("Candidate depth per modality:", data["per_modality_top_k"])
    print("Final Top-K:", TOP_K)

    output_details = []
    baseline_match_count = 0

    for q in source_details:
        methods = {}

        for name, weights in EXPERIMENTS.items():
            methods[name] = fuse_candidates(
                q,
                weights["speech"],
                weights["ocr"],
            )

        # Verify that reconstructed 1.0/1.0 ranking matches saved RRF.
        reconstructed_ids = [
            x["id"] for x in methods["baseline_1.0_1.0"]
        ]
        saved_ids = [
            x["id"] for x in q["rrf_fusion"]
        ]

        if reconstructed_ids == saved_ids:
            baseline_match_count += 1

        output_details.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "gold_speech_ids": q["gold_speech_ids"],
            "gold_ocr_ids": q["gold_ocr_ids"],
            "methods": methods,
            "saved_original_rrf_ids": saved_ids,
            "reconstructed_baseline_ids": reconstructed_ids,
            "baseline_ranking_matches_saved": (
                reconstructed_ids == saved_ids
            ),
        })

    print(
        "\nBaseline ranking matches saved original RRF:",
        f"{baseline_match_count}/{len(source_details)}",
    )

    common_ids = {
        q["question_id"]
        for q in source_details
        if q["gold_speech_ids"] and q["gold_ocr_ids"]
    }

    all_metrics = {}
    common_metrics = {}

    for method_name in EXPERIMENTS:
        all_metrics[method_name] = calculate_metrics(
            output_details,
            method_name,
        )
        common_metrics[method_name] = calculate_metrics(
            output_details,
            method_name,
            common_ids,
        )

    print("\n" + "=" * 72)
    print("ALL SIX QUESTIONS")
    print("=" * 72)

    for name, metrics in all_metrics.items():
        print(f"\n[{name}]")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:22s}: {value:.4f}")
            else:
                print(f"{key:22s}: {value}")

    print("\n" + "=" * 72)
    print("COMMON QUESTIONS: VQ01, VQ03, VQ04")
    print("=" * 72)

    for name, metrics in common_metrics.items():
        print(f"\n[{name}]")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:22s}: {value:.4f}")
            else:
                print(f"{key:22s}: {value}")

    result = {
        "experiment": "Weighted RRF comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT),
        "rrf_constant": RRF_CONSTANT,
        "candidate_depth_per_modality": data["per_modality_top_k"],
        "final_top_k": TOP_K,
        "weights": EXPERIMENTS,
        "baseline_ranking_matches_saved": baseline_match_count,
        "question_count": len(source_details),
        "common_question_ids": sorted(common_ids),
        "metrics_all_questions": all_metrics,
        "metrics_common_questions": common_metrics,
        "details": output_details,
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    fields = [
        "method", "speech_weight", "ocr_weight",
        "eligible_questions", "total_gold_evidence",
        "recall@1", "recall@3", "recall@5",
        "hitrate@1", "hitrate@3", "hitrate@5", "mrr",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for name, weights in EXPERIMENTS.items():
            writer.writerow({
                "method": name,
                "speech_weight": weights["speech"],
                "ocr_weight": weights["ocr"],
                **all_metrics[name],
            })

    lines = [
        "# Weighted RRF Evaluation Report",
        "",
        "## Experimental Setup",
        "",
        f"- Questions: {len(source_details)}",
        f"- RRF constant: {RRF_CONSTANT}",
        f"- Candidate depth per modality: {data['per_modality_top_k']}",
        f"- Final Top-K: {TOP_K}",
        "- Candidate lists are reused from the original evaluation JSON.",
        "- No embedding inference or ChromaDB query is performed.",
        "",
        "## Results: All Six Questions",
        "",
        "| Method | Speech W | OCR W | Recall@3 | Recall@5 | HitRate@3 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name, weights in EXPERIMENTS.items():
        m = all_metrics[name]
        lines.append(
            f"| {name} | {weights['speech']:.1f} | "
            f"{weights['ocr']:.1f} | {m['recall@3']:.4f} | "
            f"{m['recall@5']:.4f} | {m['hitrate@3']:.4f} | "
            f"{m['mrr']:.4f} |"
        )

    lines += [
        "",
        "## Results: Common Questions (VQ01, VQ03, VQ04)",
        "",
        "| Method | Recall@3 | Recall@5 | HitRate@3 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]

    for name in EXPERIMENTS:
        m = common_metrics[name]
        lines.append(
            f"| {name} | {m['recall@3']:.4f} | "
            f"{m['recall@5']:.4f} | {m['hitrate@3']:.4f} | "
            f"{m['mrr']:.4f} |"
        )

    lines += [
        "",
        "## Baseline Verification",
        "",
        f"- Reconstructed baseline ranking matches saved original RRF for "
        f"{baseline_match_count}/{len(source_details)} questions.",
        "",
        "## Limitations",
        "",
        "- Only six manually annotated questions are included.",
        "- Results depend on the current candidate lists and Ground Truth.",
        "- Weighted RRF changes ranking weights, not the underlying retrieval candidates.",
        "- The two modalities originate from separate video sources; their timestamps "
        "are not a shared timeline.",
        "",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 72)
    print("FILES SAVED")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)
    print(OUTPUT_MD)
    print("=" * 72)


if __name__ == "__main__":
    main()
