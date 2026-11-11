"""
Build and ingest the Smart3C FAQ knowledge base.

Default mode:
    python src/ingest.py

Actual ingestion:
    python src/ingest.py --execute
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
from vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DB_PATH,
    VectorStore,
)


TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.jsonl"
EXPECTED_RECORDS = 120
EXPECTED_DIMENSION = 1024

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
    """Load and validate the training QA records."""
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Train dataset not found: {TRAIN_PATH}")

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
                    f"Record at line {line_number} must be a JSON object."
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

    if any(not isinstance(item_id, str) or not item_id.strip()
           for item_id in ids):
        raise ValueError("Every record must have a non-empty string ID.")

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate QA IDs found in the training dataset.")

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
                    f"{record['id']}: field '{field}' must be non-empty text."
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
                f"{record['id']}: every keyword must be non-empty text."
            )

    return records


def build_documents(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Convert QA records into searchable documents and Chroma metadata."""
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


def check_collection_is_new() -> None:
    """Stop if the target collection already exists."""
    import chromadb

    client = chromadb.PersistentClient(path=str(DEFAULT_DB_PATH))
    existing_names = {
        collection.name for collection in client.list_collections()
    }

    if DEFAULT_COLLECTION_NAME in existing_names:
        collection = client.get_collection(DEFAULT_COLLECTION_NAME)
        raise RuntimeError(
            f"Collection '{DEFAULT_COLLECTION_NAME}' already exists "
            f"with {collection.count()} documents. "
            "Refusing to ingest and risk duplicate data."
        )


def preview(
    records: list[dict[str, Any]],
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    """Print a preview without loading the embedding model or writing data."""
    print("===== INGESTION DRY RUN =====")
    print(f"Dataset: {TRAIN_PATH}")
    print(f"Records: {len(records)}")
    print(f"Unique IDs: {len(set(ids))}")
    print(f"Collection: {DEFAULT_COLLECTION_NAME}")
    print(f"Database: {DEFAULT_DB_PATH}")
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

    print("\nDRY RUN PASS: No embeddings generated and no documents inserted.")
    print("To perform the actual ingestion, run:")
    print("python src/ingest.py --execute")


def execute_ingestion(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    """Generate embeddings and insert all FAQ records into ChromaDB."""
    check_collection_is_new()

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

    print("\n===== 3. INSERT INTO CHROMADB =====")
    store = VectorStore(
        persist_directory=DEFAULT_DB_PATH,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    if store.count() != 0:
        raise RuntimeError(
            "Target collection is not empty. Refusing to insert."
        )

    inserted = store.add_documents(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("Inserted:", inserted)
    print("Collection count:", store.count())

    if inserted != EXPECTED_RECORDS or store.count() != EXPECTED_RECORDS:
        raise RuntimeError(
            "Post-ingestion count verification failed. "
            "Do not rerun ingestion automatically."
        )

    print("\n===== 4. VERIFY STORED IDS =====")
    stored = store.collection.get(
        include=["metadatas"]
    )
    stored_ids = set(stored["ids"])

    if stored_ids != set(ids):
        raise RuntimeError(
            "Stored IDs do not match the training dataset IDs."
        )

    print("Stored unique IDs:", len(stored_ids))
    print("ID verification: PASS")

    print("\n===== INGESTION COMPLETE =====")
    print(f"Total documents stored: {store.count()}")
    print(f"Collection: {DEFAULT_COLLECTION_NAME}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Smart3C training FAQs into ChromaDB."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Generate embeddings and write documents to ChromaDB.",
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
