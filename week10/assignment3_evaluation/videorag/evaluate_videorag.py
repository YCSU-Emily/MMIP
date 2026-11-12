import sys
import json
import csv
from pathlib import Path
from datetime import datetime, timezone

PROJECT = Path("week10/assignment2_videorag").resolve()
EVAL_DIR = Path("week10/assignment3_evaluation/videorag").resolve()

sys.path.insert(0, str(PROJECT / "src"))

from retriever import VideoRetriever


# ============================================================
# 1. Load manually annotated Ground Truth
# ============================================================

ground_truth_path = EVAL_DIR / "video_ground_truth.json"

ground_truth = json.loads(
    ground_truth_path.read_text(encoding="utf-8")
)

questions = ground_truth["questions"]

print("=" * 70)
print("VideoRAG Retrieval Evaluation")
print("=" * 70)
print("Ground Truth questions:", len(questions))


# ============================================================
# 2. Load existing VideoRAG retriever and ChromaDB
# ============================================================

retriever = VideoRetriever(per_modality_k=8)

print("ChromaDB collection:", retriever.collection.name)
print("Corpus records:", retriever.collection.count())


# ============================================================
# 3. Retrieval functions
# ============================================================

def encode_query(query):
    return retriever.model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).tolist()


def retrieve_single_modality(query, modality, top_k=8):
    """Retrieve evidence from one modality only."""

    embedding = encode_query(query)

    result = retriever.collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        where={"modality": modality},
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    ranked = []

    for rank, (doc_id, document, metadata, distance) in enumerate(
        zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0]
        ),
        start=1
    ):
        ranked.append({
            "id": doc_id,
            "text": document,
            "metadata": metadata,
            "rank": rank,
            "distance": float(distance)
        })

    return ranked


def retrieve_fusion(query, top_k=8):
    """Use the original VideoRetriever RRF implementation."""

    results = retriever.retrieve(
        query,
        final_k=top_k
    )

    ranked = []

    for rank, item in enumerate(results, start=1):
        ranked.append({
            "id": item["id"],
            "text": item["text"],
            "metadata": item["metadata"],
            "rank": rank,
            "rrf_score": float(item["rrf_score"]),
            "distance": float(item["distance"])
        })

    return ranked


# ============================================================
# 4. Metric calculation
# ============================================================

def calculate_metrics(records):
    """
    Evidence Recall@K:
        Retrieved relevant evidence / all gold evidence

    HitRate@K:
        Questions with at least one relevant result / eligible questions

    MRR:
        Mean reciprocal rank of first relevant evidence.
    """

    output = {}

    for k in [1, 3, 5]:
        total_gold = 0
        total_retrieved_gold = 0
        hit_questions = 0

        for record in records:
            gold_ids = set(record["gold_ids"])
            retrieved_ids = [
                item["id"] for item in record["retrieved"][:k]
            ]

            total_gold += len(gold_ids)

            found = gold_ids.intersection(retrieved_ids)
            total_retrieved_gold += len(found)

            if found:
                hit_questions += 1

        eligible_questions = len(records)

        output[f"recall@{k}"] = (
            total_retrieved_gold / total_gold
            if total_gold else None
        )

        output[f"hitrate@{k}"] = (
            hit_questions / eligible_questions
            if eligible_questions else None
        )

    reciprocal_ranks = []

    for record in records:
        gold_ids = set(record["gold_ids"])

        reciprocal_rank = 0.0

        for rank, item in enumerate(record["retrieved"], start=1):
            if item["id"] in gold_ids:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    output["mrr"] = (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks else None
    )

    output["eligible_questions"] = len(records)
    output["total_gold_evidence"] = sum(
        len(record["gold_ids"]) for record in records
    )

    return output


# ============================================================
# 5. Run three retrieval experiments
# ============================================================

experiments = {
    "speech_only": [],
    "ocr_only": [],
    "rrf_fusion": []
}

detailed_results = []

for index, question in enumerate(questions, start=1):

    question_id = question["id"]
    query = question["question"]

    print("\n" + "-" * 70)
    print(f"{question_id}: {query}")

    speech_gold = question["gold_speech_ids"]
    ocr_gold = question["gold_ocr_ids"]
    fusion_gold = sorted(set(speech_gold + ocr_gold))

    speech_results = retrieve_single_modality(
        query, "speech", top_k=8
    )

    ocr_results = retrieve_single_modality(
        query, "visual_ocr", top_k=8
    )

    fusion_results = retrieve_fusion(
        query, top_k=8
    )

    candidates = [
        ("speech_only", speech_gold, speech_results),
        ("ocr_only", ocr_gold, ocr_results),
        ("rrf_fusion", fusion_gold, fusion_results)
    ]

    for method, gold_ids, retrieved in candidates:

        if not gold_ids:
            continue

        record = {
            "question_id": question_id,
            "question": query,
            "gold_ids": gold_ids,
            "retrieved": retrieved
        }

        experiments[method].append(record)

        retrieved_ids = [item["id"] for item in retrieved]
        found = sorted(set(gold_ids).intersection(retrieved_ids))

        print(
            f"{method:12s} | "
            f"Gold={gold_ids} | "
            f"Found={found}"
        )

    detailed_results.append({
        "question_id": question_id,
        "question": query,
        "gold_speech_ids": speech_gold,
        "gold_ocr_ids": ocr_gold,
        "speech_only": speech_results,
        "ocr_only": ocr_results,
        "rrf_fusion": fusion_results
    })


# ============================================================
# 6. Calculate metrics
# ============================================================

summary = {
    method: calculate_metrics(records)
    for method, records in experiments.items()
}

print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)

for method, metrics in summary.items():

    print(f"\n[{method}]")
    print("Eligible questions:", metrics["eligible_questions"])

    for key in [
        "recall@1",
        "recall@3",
        "recall@5",
        "hitrate@1",
        "hitrate@3",
        "hitrate@5",
        "mrr"
    ]:
        value = metrics[key]

        if value is None:
            print(f"{key:12s}: N/A")
        else:
            print(f"{key:12s}: {value:.4f}")


# ============================================================
# 7. Save results
# ============================================================

timestamp = datetime.now(timezone.utc).isoformat()

report = {
    "experiment": "VideoRAG Speech OCR RRF Retrieval Evaluation",
    "created_at_utc": timestamp,
    "embedding_model": "BAAI/bge-m3",
    "collection": retriever.collection.name,
    "corpus_records": retriever.collection.count(),
    "rrf_constant": 60,
    "per_modality_top_k": 8,
    "final_top_k": 8,
    "metric_definitions": {
        "recall_at_k": "Relevant retrieved evidence divided by total gold evidence.",
        "hitrate_at_k": "Fraction of eligible questions with at least one gold evidence in top K.",
        "mrr": "Mean reciprocal rank of the first retrieved gold evidence."
    },
    "ground_truth_note": ground_truth["annotation_note"],
    "results": summary,
    "details": detailed_results
}

json_path = EVAL_DIR / "videorag_evaluation_results.json"

json_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8"
)


# CSV summary
csv_path = EVAL_DIR / "videorag_evaluation_summary.csv"

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:

    fields = [
        "method",
        "eligible_questions",
        "total_gold_evidence",
        "recall@1",
        "recall@3",
        "recall@5",
        "hitrate@1",
        "hitrate@3",
        "hitrate@5",
        "mrr"
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for method, metrics in summary.items():
        writer.writerow({
            "method": method,
            **metrics
        })


# Markdown report
md_path = EVAL_DIR / "videorag_evaluation_report.md"

lines = [
    "# VideoRAG Retrieval Evaluation",
    "",
    f"- Corpus records: {retriever.collection.count()}",
    f"- Ground Truth questions: {len(questions)}",
    "- Embedding model: BAAI/bge-m3",
    "- Fusion: Reciprocal Rank Fusion (RRF), constant=60",
    "- Candidate depth: 8 per modality",
    "",
    "## Evaluation Metrics",
    "",
    "- Evidence Recall@K: retrieved gold evidence / total gold evidence.",
    "- HitRate@K: proportion of eligible questions with at least one gold result.",
    "- MRR: mean reciprocal rank of the first relevant evidence.",
    "",
    "## Results",
    "",
    "| Method | Eligible Q | Recall@1 | Recall@3 | Recall@5 | HitRate@3 | MRR |",
    "|---|---:|---:|---:|---:|---:|---:|"
]

for method, metrics in summary.items():

    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"

    lines.append(
        f"| {method} "
        f"| {metrics['eligible_questions']} "
        f"| {fmt(metrics['recall@1'])} "
        f"| {fmt(metrics['recall@3'])} "
        f"| {fmt(metrics['recall@5'])} "
        f"| {fmt(metrics['hitrate@3'])} "
        f"| {fmt(metrics['mrr'])} |"
    )

lines.extend([
    "",
    "## Annotation Notes",
    "",
    ground_truth["annotation_note"],
    "",
    "Speech-only and OCR-only metrics use only questions with gold evidence in that modality.",
    "RRF Fusion uses the union of Speech and OCR gold evidence.",
    "The speech and OCR recordings are separate video sources; their timestamps are not treated as one shared timeline.",
    ""
])

md_path.write_text("\n".join(lines), encoding="utf-8")

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)
print(json_path)
print(csv_path)
print(md_path)
print("\nEvaluation completed.")
