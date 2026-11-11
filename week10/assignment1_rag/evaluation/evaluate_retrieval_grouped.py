"""
Evaluate Smart3C retrieval on the Group Split V3 held-out test set.

Metrics:
- Top-1 category accuracy
- Top-3 category hit rate

Category metrics are diagnostic proxies, not human relevance judgments.
"""

import json
from pathlib import Path
from typing import Any

import chromadb

from src.retriever import FAQRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = PROJECT_ROOT / "data" / "splits_grouped" / "test.jsonl"
DB_PATH = PROJECT_ROOT / "vector_db_grouped"
COLLECTION_NAME = "smart3c_faq_grouped"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "retrieval_eval_grouped_v1.json"
TOP_K = 3


def load_test_records() -> list[dict[str, Any]]:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test set not found: {TEST_PATH}")

    with TEST_PATH.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if len(records) != 16:
        raise ValueError(f"Expected 16 test records, got {len(records)}.")

    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs found in test set.")

    return records


def verify_source_separation(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Grouped database not found: {DB_PATH}")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    stored = collection.get(include=["metadatas"])
    train_sources = {
        metadata["source_id"]
        for metadata in stored["metadatas"]
        if metadata and metadata.get("source_id")
    }
    test_sources = {
        record["source_id"]
        for record in records
    }

    overlap = train_sources & test_sources

    print("===== SOURCE SEPARATION CHECK =====")
    print("Train source groups:", len(train_sources))
    print("Test source groups:", len(test_sources))
    print("Overlapping source groups:", len(overlap))

    if overlap:
        raise RuntimeError(
            f"Source leakage detected: {sorted(overlap)}"
        )

    print("PASS: Train and Test source groups are disjoint.")
    return {
        "train_source_groups": len(train_sources),
        "test_source_groups": len(test_sources),
        "overlapping_source_groups": len(overlap),
    }


def main() -> None:
    if REPORT_PATH.exists():
        raise FileExistsError(
            f"STOP: report already exists: {REPORT_PATH}"
        )

    records = load_test_records()
    separation = verify_source_separation(records)

    print("\n===== LOAD RETRIEVER =====")
    retriever = FAQRetriever()
    print("Database collection:", COLLECTION_NAME)
    print("Database count:", retriever.vector_store.count())
    print("Test queries:", len(records))
    print("Top-K:", TOP_K)

    results = []
    top1_correct = 0
    top3_hit = 0

    print("\n===== RUN TEST QUERIES =====")

    for index, record in enumerate(records, start=1):
        retrieved = retriever.retrieve(
            query=record["question"],
            top_k=TOP_K,
        )

        expected_category = record["category"]
        predicted_category = (
            retrieved[0]["category"] if retrieved else ""
        )

        is_top1_correct = predicted_category == expected_category
        is_top3_hit = any(
            item["category"] == expected_category
            for item in retrieved
        )

        top1_correct += int(is_top1_correct)
        top3_hit += int(is_top3_hit)

        top_results = [
            {
                "rank": item["rank"],
                "id": item["id"],
                "category": item["category"],
                "similarity": item["similarity"],
                "source_id": item["source_id"],
                "source_title": item["source_title"],
                "source_url": item["source_url"],
            }
            for item in retrieved
        ]

        result = {
            "query_id": record["id"],
            "query": record["question"],
            "expected_category": expected_category,
            "top1_category": predicted_category,
            "top1_correct": is_top1_correct,
            "top3_category_hit": is_top3_hit,
            "top_results": top_results,
        }
        results.append(result)

        status = "TOP1" if is_top1_correct else (
            "TOP3" if is_top3_hit else "MISS"
        )
        top1_text = top_results[0]["id"] if top_results else "NONE"
        top1_similarity = (
            top_results[0]["similarity"] if top_results else 0.0
        )

        print(
            f"{index:02d}/{len(records)} | "
            f"{record['id']} | "
            f"expected={expected_category} | "
            f"top1={top1_text} | "
            f"similarity={top1_similarity:.4f} | "
            f"{status}"
        )

    total = len(records)
    metrics = {
        "total_queries": total,
        "top1_category_correct": top1_correct,
        "top1_category_accuracy": top1_correct / total,
        "top3_category_hit": top3_hit,
        "top3_category_hit_rate": top3_hit / total,
        "top_k": TOP_K,
    }

    report = {
        "experiment": "Smart3C Group Split V3 Retrieval Evaluation",
        "database": str(DB_PATH),
        "collection": COLLECTION_NAME,
        "test_dataset": str(TEST_PATH),
        "source_separation": separation,
        "metrics": metrics,
        "per_query_results": results,
        "metric_note": (
            "Category accuracy and category hit rate are diagnostic "
            "proxies. They do not replace human semantic relevance "
            "judgments."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\n===== EVALUATION SUMMARY =====")
    print(
        f"Top-1 Category Accuracy: "
        f"{top1_correct}/{total} = "
        f"{metrics['top1_category_accuracy']:.2%}"
    )
    print(
        f"Top-3 Category Hit Rate: "
        f"{top3_hit}/{total} = "
        f"{metrics['top3_category_hit_rate']:.2%}"
    )
    print("Report:", REPORT_PATH)
    print("\nGROUPED RETRIEVAL EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
