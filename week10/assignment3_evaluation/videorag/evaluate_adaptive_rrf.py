import sys
import json
import csv
from pathlib import Path
from datetime import datetime, timezone

PROJECT = Path("week10/assignment2_videorag").resolve()
EVAL_DIR = Path("week10/assignment3_evaluation/videorag").resolve()

GROUND_TRUTH = EVAL_DIR / "video_ground_truth_v2.json"

OUTPUT_JSON = EVAL_DIR / "adaptive_rrf_results.json"
OUTPUT_CSV = EVAL_DIR / "adaptive_rrf_summary.csv"
OUTPUT_MD = EVAL_DIR / "adaptive_rrf_report.md"

RRF_CONSTANT = 60
PER_MODALITY_TOP_K = 8
FINAL_TOP_K = 8

sys.path.insert(0, str(PROJECT / "src"))
from retriever import VideoRetriever


# ------------------------------------------------------------
# 1. Load V2 Ground Truth
# ------------------------------------------------------------

ground_truth = json.loads(
    GROUND_TRUTH.read_text(encoding="utf-8")
)

questions = ground_truth["questions"]

if not questions:
    raise ValueError("Ground Truth contains no questions.")

for q in questions:
    if q["intent"] not in ("speech", "visual"):
        raise ValueError(
            f"Invalid intent in {q['id']}: {q['intent']}"
        )

print("=" * 72)
print("VideoRAG Query-Adaptive RRF Evaluation")
print("=" * 72)
print("Questions:", len(questions))
print("Ground Truth:", GROUND_TRUTH)


# ------------------------------------------------------------
# 2. Initialize the original VideoRetriever
# ------------------------------------------------------------

retriever = VideoRetriever(
    per_modality_k=PER_MODALITY_TOP_K
)

print("Collection:", retriever.collection.name)
print("Corpus records:", retriever.collection.count())


# ------------------------------------------------------------
# 3. Retrieve Speech and OCR candidates independently
# ------------------------------------------------------------

def encode_query(query):
    return retriever.model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    ).tolist()


def retrieve_modality(query_embedding, modality):
    result = retriever.collection.query(
        query_embeddings=query_embedding,
        n_results=PER_MODALITY_TOP_K,
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
            "distance": float(distance),
            "modality": modality
        })

    return ranked


# ------------------------------------------------------------
# 4. Fusion methods
# ------------------------------------------------------------

def weighted_rrf(speech_items, ocr_items,
                 speech_weight, ocr_weight):
    """
    Weighted Reciprocal Rank Fusion:

    WRRF(d) =
        speech_weight / (60 + speech_rank)
        + ocr_weight / (60 + ocr_rank)

    Each evidence record belongs to one modality.
    """

    fused = {}

    for modality, items, weight in [
        ("speech", speech_items, speech_weight),
        ("visual_ocr", ocr_items, ocr_weight)
    ]:
        for item in items:
            score = weight / (RRF_CONSTANT + item["rank"])

            fused[item["id"]] = {
                **item,
                "rrf_score": float(score)
            }

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -item["rrf_score"],
            item["distance"]
        )
    )

    return ranked[:FINAL_TOP_K]


def get_experiment_weights(intent, method):
    if method == "baseline":
        return 1.0, 1.0

    if method == "fixed_weighted":
        return 1.5, 1.0

    if method == "adaptive":
        if intent == "visual":
            return 1.0, 1.5
        return 1.5, 1.0

    raise ValueError(f"Unknown method: {method}")


METHODS = [
    "baseline",
    "fixed_weighted",
    "adaptive"
]


# ------------------------------------------------------------
# 5. Evaluate all questions
# ------------------------------------------------------------

details = []

for index, q in enumerate(questions, start=1):
    print(
        f"[{index}/{len(questions)}] "
        f"{q['id']}: {q['question']}"
    )

    query_embedding = encode_query(q["question"])

    speech_items = retrieve_modality(
        query_embedding,
        "speech"
    )

    ocr_items = retrieve_modality(
        query_embedding,
        "visual_ocr"
    )

    # Original VideoRetriever baseline verification.
    original_results = retriever.retrieve(
        q["question"],
        final_k=FINAL_TOP_K
    )

    original_ids = [
        item["id"] for item in original_results
    ]

    baseline = weighted_rrf(
        speech_items,
        ocr_items,
        1.0,
        1.0
    )

    reconstructed_ids = [
        item["id"] for item in baseline
    ]

    if reconstructed_ids != original_ids:
        raise RuntimeError(
            f"Baseline mismatch for {q['id']}!\n"
            f"Reconstructed: {reconstructed_ids}\n"
            f"Original:      {original_ids}"
        )

    gold_speech = set(q["gold_speech_ids"])
    gold_ocr = set(q["gold_ocr_ids"])
    gold_ids = gold_speech | gold_ocr

    if not gold_ids:
        raise ValueError(
            f"No Ground Truth evidence for {q['id']}"
        )

    method_results = {}

    for method in METHODS:
        speech_weight, ocr_weight = get_experiment_weights(
            q["intent"],
            method
        )

        ranked = weighted_rrf(
            speech_items,
            ocr_items,
            speech_weight,
            ocr_weight
        )

        method_results[method] = {
            "speech_weight": speech_weight,
            "ocr_weight": ocr_weight,
            "retrieved": ranked,
            "retrieved_ids": [
                item["id"] for item in ranked
            ]
        }

    details.append({
        "question_id": q["id"],
        "question": q["question"],
        "intent": q["intent"],
        "gold_speech_ids": sorted(gold_speech),
        "gold_ocr_ids": sorted(gold_ocr),
        "gold_ids": sorted(gold_ids),
        "speech_candidates": speech_items,
        "ocr_candidates": ocr_items,
        "original_baseline_verified": True,
        "methods": method_results
    })


# ------------------------------------------------------------
# 6. Metric calculation
# ------------------------------------------------------------

def calculate_metrics(records, method):
    total_gold = 0
    output = {}

    for k in [1, 3, 5]:
        retrieved_gold = 0
        hit_questions = 0

        for record in records:
            gold = set(record["gold_ids"])
            retrieved = record["methods"][method][
                "retrieved_ids"
            ][:k]

            found = gold.intersection(retrieved)

            total_gold += 0 if k != 1 else len(gold)
            retrieved_gold += len(found)

            if found:
                hit_questions += 1

        output[f"recall@{k}"] = (
            retrieved_gold / total_gold
            if total_gold else 0.0
        )

        output[f"hitrate@{k}"] = (
            hit_questions / len(records)
            if records else 0.0
        )

    reciprocal_ranks = []

    for record in records:
        gold = set(record["gold_ids"])
        retrieved = record["methods"][method]["retrieved_ids"]

        rr = 0.0

        for rank, evidence_id in enumerate(retrieved, start=1):
            if evidence_id in gold:
                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

    output["mrr"] = (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks else 0.0
    )

    output["eligible_questions"] = len(records)
    output["total_gold_evidence"] = sum(
        len(record["gold_ids"]) for record in records
    )

    return output


summaries = {}

for method in METHODS:
    summaries[method] = calculate_metrics(
        details,
        method
    )


# ------------------------------------------------------------
# 7. Save JSON
# ------------------------------------------------------------

payload = {
    "experiment": "VideoRAG Query-Adaptive RRF Evaluation",
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "embedding_model": "BAAI/bge-m3",
    "collection": retriever.collection.name,
    "corpus_records": retriever.collection.count(),
    "rrf_constant": RRF_CONSTANT,
    "per_modality_top_k": PER_MODALITY_TOP_K,
    "final_top_k": FINAL_TOP_K,
    "intent_assignment": (
        "Manually annotated intent labels in V2 Ground Truth. "
        "This experiment evaluates rule-based adaptive fusion, "
        "not an automatic intent classifier."
    ),
    "method_definitions": {
        "baseline": {
            "speech_weight": 1.0,
            "ocr_weight": 1.0
        },
        "fixed_weighted": {
            "speech_weight": 1.5,
            "ocr_weight": 1.0
        },
        "adaptive": {
            "visual_intent": {
                "speech_weight": 1.0,
                "ocr_weight": 1.5
            },
            "speech_intent": {
                "speech_weight": 1.5,
                "ocr_weight": 1.0
            }
        }
    },
    "metric_definitions": {
        "recall@k": (
            "Number of retrieved gold evidence IDs in top K "
            "divided by total gold evidence IDs."
        ),
        "hitrate@k": (
            "Fraction of questions with at least one gold "
            "evidence ID in top K."
        ),
        "mrr": (
            "Mean reciprocal rank of the first retrieved "
            "gold evidence."
        )
    },
    "baseline_verification": {
        "verified_questions": sum(
            1 for d in details
            if d["original_baseline_verified"]
        ),
        "total_questions": len(details)
    },
    "summary": summaries,
    "details": details
}

OUTPUT_JSON.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 8. Save CSV
# ------------------------------------------------------------

csv_fields = [
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

with OUTPUT_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=csv_fields
    )
    writer.writeheader()

    for method, metrics in summaries.items():
        writer.writerow({
            "method": method,
            **metrics
        })


# ------------------------------------------------------------
# 9. Save Markdown report
# ------------------------------------------------------------

labels = {
    "baseline": "Baseline RRF",
    "fixed_weighted": "Fixed Weighted RRF",
    "adaptive": "Query-Adaptive RRF"
}

lines = [
    "# VideoRAG Query-Adaptive RRF Evaluation",
    "",
    "## Experimental Setup",
    "",
    f"- Questions: {len(questions)}",
    f"- Corpus records: {retriever.collection.count()}",
    f"- RRF constant: {RRF_CONSTANT}",
    f"- Candidate depth per modality: {PER_MODALITY_TOP_K}",
    f"- Final Top-K: {FINAL_TOP_K}",
    "- Speech and OCR candidates are retrieved independently.",
    "- All methods use identical candidate lists for each query.",
    "- Intent labels are manually annotated, not automatically predicted.",
    "- Speech and OCR originate from separate source videos.",
    "",
    "## Results",
    "",
    "| Method | Recall@1 | Recall@3 | Recall@5 | HitRate@3 | MRR |",
    "|---|---:|---:|---:|---:|---:|"
]

for method, metrics in summaries.items():
    lines.append(
        f"| {labels[method]} "
        f"| {metrics['recall@1']:.4f} "
        f"| {metrics['recall@3']:.4f} "
        f"| {metrics['recall@5']:.4f} "
        f"| {metrics['hitrate@3']:.4f} "
        f"| {metrics['mrr']:.4f} |"
    )

lines += [
    "",
    "## Baseline Verification",
    "",
    f"- Reconstructed baseline matches the original VideoRetriever for "
    f"{payload['baseline_verification']['verified_questions']}/"
    f"{payload['baseline_verification']['total_questions']} questions.",
    "",
    "## Per-Question Adaptive Retrieval",
    "",
    "| Question | Intent | Speech W | OCR W | First Gold Rank |",
    "|---|---|---:|---:|---:|"
]

for record in details:
    adaptive = record["methods"]["adaptive"]
    gold = set(record["gold_ids"])
    first_gold_rank = next(
        (
            rank
            for rank, item in enumerate(
                adaptive["retrieved"],
                start=1
            )
            if item["id"] in gold
        ),
        None
    )

    lines.append(
        f"| {record['question_id']} "
        f"| {record['intent']} "
        f"| {adaptive['speech_weight']:.1f} "
        f"| {adaptive['ocr_weight']:.1f} "
        f"| {first_gold_rank if first_gold_rank else 'Not found'} |"
    )

lines += [
    "",
    "## Limitations",
    "",
    "- Only six additional questions are evaluated.",
    "- Questions are new, but they use the existing video corpus.",
    "- Ground Truth evidence IDs are manually annotated.",
    "- Intent labels are provided manually; automatic intent classification "
    "has not been evaluated.",
    "- The experiment evaluates retrieval ranking, not answer-generation quality.",
    "- Speech and OCR timestamps belong to separate source videos and "
    "must not be interpreted as one shared timeline.",
    ""
]

OUTPUT_MD.write_text(
    "\n".join(lines),
    encoding="utf-8"
)


# ------------------------------------------------------------
# 10. Print results
# ------------------------------------------------------------

print("\n" + "=" * 72)
print("RESULTS")
print("=" * 72)

for method, metrics in summaries.items():
    print(f"\n[{labels[method]}]")

    for key in [
        "eligible_questions",
        "total_gold_evidence",
        "recall@1",
        "recall@3",
        "recall@5",
        "hitrate@1",
        "hitrate@3",
        "hitrate@5",
        "mrr"
    ]:
        value = metrics[key]

        if isinstance(value, float):
            print(f"{key:22s}: {value:.4f}")
        else:
            print(f"{key:22s}: {value}")

print("\nBaseline verified:",
      payload["baseline_verification"])

print("\nFILES SAVED")
print(OUTPUT_JSON)
print(OUTPUT_CSV)
print(OUTPUT_MD)
print("=" * 72)
