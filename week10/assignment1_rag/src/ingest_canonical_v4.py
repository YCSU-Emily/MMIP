"""
Build an isolated ChromaDB for Canonical Document Split V4.

Dry run:
    python -m src.ingest_canonical_v4

Actual ingestion:
    python -m src.ingest_canonical_v4 --execute
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedding import TextEmbedder
from vector_store import VectorStore


TRAIN_PATH = (
    PROJECT_ROOT / "data" / "splits_canonical_v4" / "train.jsonl"
)
DB_PATH = PROJECT_ROOT / "vector_db_canonical_v4"
COLLECTION_NAME = "smart3c_faq_canonical_v4"

EXPECTED_RECORDS = 118
EXPECTED_DIMENSION = 1024
EXPECTED_CATEGORIES = 7

REQUIRED_FIELDS = {
    "id",
    "question",
    "answer",
    "keywords",
    "difficulty",
    "source_id",
    "category",
    "source_title",
    "source_publisher",
    "source_url",
    "verified_at",
}


def load_train_records() -> list[dict[str, Any]]:
    """Load and validate Canonical V4 training records."""
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(
            f"Train dataset not found: {TRAIN_PATH}"
        )

    records = []

    with TRAIN_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record at line {line_number} must be an object."
                )

            missing = REQUIRED_FIELDS - set(record)
            if missing:
                raise ValueError(
                    f"Record at line {line_number} is missing: "
                    f"{sorted(missing)}"
                )

            records.append(record)

    if len(records) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} records, got {len(records)}."
        )

    ids = [record["id"] for record in records]
    categories = {record["category"] for record in records}

    if any(
        not isinstance(item_id, str) or not item_id.strip()
        for item_id in ids
    ):
        raise ValueError("Every record must have a non-empty string ID.")

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate QA IDs found.")

    if len(categories) != EXPECTED_CATEGORIES:
        raise ValueError(
            f"Expected {EXPECTED_CATEGORIES} categories, "
            f"got {len(categories)}."
        )

    for record in records:
        for field in (
            "question",
            "answer",
            "difficulty",
            "source_id",
            "category",
            "source_title",
            "source_publisher",
            "source_url",
            "verified_at",
        ):
            value = record[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{record['id']}: '{field}' must be non-empty text."
                )

        if not isinstance(record["keywords"], list):
            raise ValueError(
                f"{record['id']}: keywords must be a list."
            )

        if not all(
            isinstance(keyword, str) and keyword.strip()
            for keyword in record["keywords"]
        ):
            raise ValueError(
                f"{record['id']}: invalid keyword."
            )

    return records


def build_documents(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Convert QA records into ChromaDB documents and metadata."""
    ids = []
    documents = []
    metadatas = []

    for record in records:
        keywords = ", ".join(record["keywords"])

        document = (
            f"Question: {record['question']}\n"
            f"Answer: {record['answer']}\n"
            f"Keywords: {keywords}\n"
            f"Category: {record['category']}"
        )

        metadata = {
            "category": record["category"],
            "difficulty": record["difficulty"],
            "source_id": record["source_id"],
            "source_title": record["source_title"],
            "source_publisher": record["source_publisher"],
            "source_url": record["source_url"],
            "verified_at": record["verified_at"],
            "keywords": keywords,
        }

        ids.append(record["id"])
        documents.append(document)
        metadatas.append(metadata)

    return ids, documents, metadatas


def preview(
    records: list[dict[str, Any]],
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    """Show ingestion information without writing a database."""
    print("===== CANONICAL V4 INGESTION DRY RUN =====")
    print(f"Dataset: {TRAIN_PATH}")
    print(f"Records: {len(records)}")
    print(f"Unique IDs: {len(set(ids))}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database: {DB_PATH}")
    print(f"Embedding model: BAAI/bge-m3")
    print(f"Embedding dimension: {EXPECTED_DIMENSION}")

    print("\n===== CATEGORY DISTRIBUTION =====")
    for category, count in sorted(
        Counter(record["category"] for record in records).items()
    ):
        print(f"{category}: {count}")

    print("\n===== FIRST DOCUMENT PREVIEW =====")
    print("ID:", ids[0])
    print(documents[0])
    print("\nMetadata:")
    print(json.dumps(metadatas[0], ensure_ascii=False, indent=2))

    print("\nDRY RUN PASS")
    print("No embeddings generated. No database created.")
    print("\nTo perform actual ingestion, run:")
    print("python -m src.ingest_canonical_v4 --execute")


def execute_ingestion(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    """Generate embeddings and insert them into a new database."""
    if DB_PATH.exists():
        raise FileExistsError(
            f"STOP: target database already exists: {DB_PATH}"
        )

    print("===== 1. LOAD EMBEDDING MODEL =====")
    embedder = TextEmbedder(batch_size=16)

    dimension = embedder.get_embedding_dimension()
    if dimension != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"Expected {EXPECTED_DIMENSION} dimensions, got {dimension}."
        )

    print("Model:", embedder.model_name)
    print("Device:", embedder.device)
    print("Embedding dimension:", dimension)

    print("\n===== 2. GENERATE DOCUMENT EMBEDDINGS =====")
    embeddings = embedder.embed_documents(documents)

    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)

    if embeddings.shape != (len(ids), EXPECTED_DIMENSION):
        raise RuntimeError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    print("\n===== 3. CREATE NEW DATABASE =====")
    DB_PATH.mkdir(parents=True, exist_ok=False)

    store = VectorStore(
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    if store.count() != 0:
        raise RuntimeError(
            "New collection is not empty. Refusing to insert."
        )

    print("\n===== 4. INSERT DOCUMENTS =====")
    inserted = store.add_documents(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    final_count = store.count()

    print("Inserted:", inserted)
    print("Final collection count:", final_count)

    if inserted != EXPECTED_RECORDS:
        raise RuntimeError("Inserted count verification failed.")

    if final_count != EXPECTED_RECORDS:
        raise RuntimeError("Final database count verification failed.")

    print("\nCANONICAL V4 INGESTION COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Canonical Document V4 vector database."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Generate embeddings and write to the new database.",
    )
    args = parser.parse_args()

    records = load_train_records()
    ids, documents, metadatas = build_documents(records)

    if args.execute:
        execute_ingestion(ids, documents, metadatas)
    else:
        preview(records, ids, documents, metadatas)


if __name__ == "__main__":
    main()
