"""
Evaluate Smart3C Canonical Document Split V4 retrieval.

Metrics:
- Top-1 category accuracy
- Top-1 / Top-3 / Top-5 category hit rate
- Exact QA ID hit rate at K
- Reference completeness
- Canonical URL document separation

Category metrics are diagnostic proxies, not human relevance judgments.
"""

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import chromadb

from src.retriever import FAQRetriever


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "splits_canonical_v4" / "train.jsonl"
TEST_PATH = ROOT / "data" / "splits_canonical_v4" / "test.jsonl"
MANIFEST_PATH = ROOT / "data" / "splits_canonical_v4" / "split_manifest.json"

DB_PATH = ROOT / "vector_db_canonical_v4"
COLLECTION_NAME = "smart3c_faq_canonical_v4"

REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "retrieval_eval_canonical_v4_v1.json"

MAX_TOP_K = 5
EVALUATION_KS = (1, 3, 5)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    return records


def canonicalize_url(url: str) -> str:
    """Keep the same canonical URL rules used by split_canonical_v4.py."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("A record has a missing source_url.")

    parts = urlsplit(url.strip())

    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        raise ValueError(f"Invalid source_url: {url}")

    scheme = parts.scheme.lower()
    host = parts.netloc.lower()

    if host.startswith("www."):
        host = host[4:]

    path = re.sub(r"/+", "/", parts.path.lower())
    segments = [segment for segment in path.split("/") if segment]

    if segments and re.fullmatch(r"[a-z]{2}-[a-z]{2}", segments[0]):
        segments = segments[1:]

    normalized_path = "/" + "/".join(segments)

    return urlunsplit((
        scheme,
        host,
        normalized_path.rstrip("/"),
        "",
        "",
    ))


def verify_data_and_separation(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> dict[str, Any]:

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)

    train_ids = {record["id"] for record in train_records}
    test_ids = {record["id"] for record in test_records}

    if len(train_ids) != len(train_records):
        raise ValueError("Duplicate QA IDs in Train.")

    if len(test_ids) != len(test_records):
        raise ValueError("Duplicate QA IDs in Test.")

    if train_ids & test_ids:
        raise RuntimeError("QA ID leakage detected between Train and Test.")

    train_urls = {
        canonicalize_url(record["source_url"])
        for record in train_records
    }
    test_urls = {
        canonicalize_url(record["source_url"])
        for record in test_records
    }

    overlap = train_urls & test_urls

    if overlap:
        raise RuntimeError(
            f"Canonical URL leakage detected: {sorted(overlap)}"
        )

    # Cross-check the manifest's canonical URL records.
    manifest_train_urls = set(
        manifest["splits"]["train"]["canonical_urls"]
    )
    manifest_test_urls = set(
        manifest["splits"]["test"]["canonical_urls"]
    )

    if train_urls != manifest_train_urls:
        raise RuntimeError("Train URLs do not match the V4 manifest.")

    if test_urls != manifest_test_urls:
        raise RuntimeError("Test URLs do not match the V4 manifest.")

    print("===== CANONICAL URL SEPARATION CHECK =====")
    print("Train QA:", len(train_records))
    print("Test QA:", len(test_records))
    print("Train canonical documents:", len(train_urls))
    print("Test canonical documents:", len(test_urls))
    print("Overlapping canonical documents:", len(overlap))
    print("PASS: Train and Test canonical documents are disjoint.")

    return {
        "train_qa_count": len(train_records),
        "test_qa_count": len(test_records),
        "train_canonical_document_count": len(train_urls),
        "test_canonical_document_count": len(test_urls),
        "overlapping_canonical_document_count": len(overlap),
        "qa_id_overlap_count": 0,
        "manifest_url_match": True,
    }


def verify_database(train_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"V4 database not found: {DB_PATH}")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    stored = collection.get(include=["metadatas"])

    stored_ids = set(stored["ids"])
    train_ids = {record["id"] for record in train_records}

    if stored_ids != train_ids:
        missing = sorted(train_ids - stored_ids)
        unexpected = sorted(stored_ids - train_ids)
        raise RuntimeError(
            f"Database QA IDs do not match Train. "
            f"Missing={missing}, unexpected={unexpected}"
        )

    stored_urls = {
        canonicalize_url(metadata["source_url"])
        for metadata in stored["metadatas"]
        if metadata and metadata.get("source_url")
    }

    train_urls = {
        canonicalize_url(record["source_url"])
        for record in train_records
    }

    if stored_urls != train_urls:
        raise RuntimeError("Database canonical URLs do not match Train.")

    print("\n===== V4 DATABASE CHECK =====")
    print("Collection:", COLLECTION_NAME)
    print("Database count:", collection.count())
    print("Train QA count:", len(train_records))
    print("Database QA IDs match Train: PASS")
    print("Database canonical URLs match Train: PASS")

    return {
        "database_count": collection.count(),
        "database_qa_ids_match_train": True,
        "database_canonical_urls_match_train": True,
    }


def main() -> None:
    if REPORT_PATH.exists():
        raise FileExistsError(
            f"STOP: report already exists: {REPORT_PATH}"
        )

    train_records = load_jsonl(TRAIN_PATH)
    test_records = load_jsonl(TEST_PATH)

    if len(train_records) != 118:
        raise ValueError(f"Expected 118 Train records, got {len(train_records)}.")

    if len(test_records) != 16:
        raise ValueError(f"Expected 16 Test records, got {len(test_records)}.")

    separation = verify_data_and_separation(
        train_records,
        test_records,
    )
    database_check = verify_database(train_records)

    print("\n===== LOAD V4 RETRIEVER =====")

    retriever = FAQRetriever(
        persist_directory=str(DB_PATH),
        collection_name=COLLECTION_NAME,
    )

    print("Database count:", retriever.vector_store.count())
    print("Test queries:", len(test_records))
    print("Evaluation K values:", EVALUATION_KS)

    metric_counts = {
        k: {
            "category_hit": 0,
            "exact_qa_id_hit": 0,
        }
        for k in EVALUATION_KS
    }

    top1_category_correct = 0
    reference_complete_count = 0
    per_query_results = []

    print("\n===== RUN V4 TEST QUERIES =====")

    for index, record in enumerate(test_records, start=1):
        retrieved = retriever.retrieve(
            query=record["question"],
            top_k=MAX_TOP_K,
        )

        expected_category = record["category"]
        expected_id = record["id"]

        top1_category = (
            retrieved[0]["category"] if retrieved else ""
        )

        is_top1_correct = top1_category == expected_category
        top1_category_correct += int(is_top1_correct)

        query_result = {
            "query_id": expected_id,
            "query": record["question"],
            "expected_category": expected_category,
            "top1_category": top1_category,
            "top1_category_correct": is_top1_correct,
            "top_results": [],
            "metrics_at_k": {},
        }

        for item in retrieved:
            query_result["top_results"].append({
                "rank": item["rank"],
                "id": item["id"],
                "category": item["category"],
                "similarity": item["similarity"],
                "source_id": item["source_id"],
                "source_title": item["source_title"],
                "source_publisher": item["source_publisher"],
                "source_url": item["source_url"],
            })

        has_complete_references = all(
            item.get("source_title")
            and item.get("source_publisher")
            and item.get("source_url")
            for item in retrieved
        )

        if retrieved and has_complete_references:
            reference_complete_count += 1

        for k in EVALUATION_KS:
            top_k_results = retrieved[:k]

            category_hit = any(
                item["category"] == expected_category
                for item in top_k_results
            )

            exact_id_hit = any(
                item["id"] == expected_id
                for item in top_k_results
            )

            metric_counts[k]["category_hit"] += int(category_hit)
            metric_counts[k]["exact_qa_id_hit"] += int(exact_id_hit)

            query_result["metrics_at_k"][str(k)] = {
                "category_hit": category_hit,
                "exact_qa_id_hit": exact_id_hit,
            }

        per_query_results.append(query_result)

        status = "TOP1" if is_top1_correct else (
            "TOP5" if metric_counts[5]["category_hit"] else "CHECK"
        )

        retrieved_ids = ", ".join(
            item["id"] for item in retrieved
        )

        top1_similarity = (
            retrieved[0]["similarity"] if retrieved else 0.0
        )

        print(
            f"{index:02d}/{len(test_records)} | "
            f"{expected_id} | "
            f"expected={expected_category} | "
            f"top1={retrieved[0]['id'] if retrieved else 'NONE'} | "
            f"similarity={top1_similarity:.4f} | "
            f"retrieved=[{retrieved_ids}]"
        )

    total = len(test_records)

    metrics = {
        "total_queries": total,
        "top1_category_accuracy": top1_category_correct / total,
        "top1_category_correct": top1_category_correct,
        "reference_complete_queries": reference_complete_count,
        "reference_completeness_rate": reference_complete_count / total,
        "by_k": {},
    }

    for k in EVALUATION_KS:
        metrics["by_k"][str(k)] = {
            "category_hit_count": metric_counts[k]["category_hit"],
            "category_hit_rate": metric_counts[k]["category_hit"] / total,
            "exact_qa_id_hit_count": metric_counts[k]["exact_qa_id_hit"],
            "exact_qa_id_hit_rate": (
                metric_counts[k]["exact_qa_id_hit"] / total
            ),
        }

    report = {
        "experiment": "Smart3C Canonical Document Split V4 Retrieval Evaluation",
        "database": str(DB_PATH),
        "collection": COLLECTION_NAME,
        "train_dataset": str(TRAIN_PATH),
        "test_dataset": str(TEST_PATH),
        "canonical_separation": separation,
        "database_integrity": database_check,
        "metrics": metrics,
        "per_query_results": per_query_results,
        "metric_note": (
            "Category hit rates are diagnostic proxies, not human "
            "semantic relevance judgments. Exact QA ID hit rate is a "
            "strict matching metric and does not imply that other "
            "retrieved answers are irrelevant. References are checked "
            "for metadata completeness, not independently fact-checked."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\n===== V4 EVALUATION SUMMARY =====")
    print(
        f"Top-1 Category Accuracy: "
        f"{top1_category_correct}/{total} = "
        f"{metrics['top1_category_accuracy']:.2%}"
    )

    for k in EVALUATION_KS:
        result = metrics["by_k"][str(k)]
        print(
            f"Top-{k} Category Hit Rate: "
            f"{result['category_hit_count']}/{total} = "
            f"{result['category_hit_rate']:.2%}"
        )
        print(
            f"Top-{k} Exact QA ID Hit Rate: "
            f"{result['exact_qa_id_hit_count']}/{total} = "
            f"{result['exact_qa_id_hit_rate']:.2%}"
        )

    print(
        f"Reference Completeness: "
        f"{reference_complete_count}/{total} = "
        f"{metrics['reference_completeness_rate']:.2%}"
    )
    print("Report:", REPORT_PATH)
    print("\nCANONICAL V4 RETRIEVAL EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
