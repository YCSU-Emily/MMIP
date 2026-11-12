from pathlib import Path
import json

import chromadb
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = ROOT / "outputs" / "video_corpus.jsonl"
DB_PATH = ROOT / "outputs" / "videorag_chroma_db"

MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
    / "5617a9f61b028005a4858fdac845db406aefb181"
)

COLLECTION_NAME = "videorag_corpus"
BATCH_SIZE = 8


def load_records():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

    records = []

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError("Video corpus is empty.")

    ids = [record["record_id"] for record in records]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate record_id found in corpus.")

    return records


def build_metadata(record):
    metadata = {
        "video_id": str(record["video_id"]),
        "modality": str(record["modality"]),
        "start_seconds": float(record["start_seconds"]),
        "end_seconds": float(record["end_seconds"]),
        "source_file": str(record["source_file"]),
    }

    # ChromaDB metadata values must be scalar types.
    for key, value in record.get("metadata", {}).items():
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value

    return metadata


def main():
    print("========== 1. LOAD VIDEO CORPUS ==========")

    records = load_records()
    print("Records:", len(records))

    print("\n========== 2. LOAD BGE-M3 ON CPU ==========")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    torch.set_num_threads(4)

    model = SentenceTransformer(
        str(MODEL_PATH),
        device="cpu",
        local_files_only=True,
    )

    print("Model loaded.")
    print("Device: CPU")

    print("\n========== 3. PREPARE CHROMADB ==========")

    DB_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("Database:", DB_PATH)
    print("Collection:", COLLECTION_NAME)
    print("Existing records:", collection.count())

    print("\n========== 4. GENERATE EMBEDDINGS AND UPSERT ==========")

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        texts = [record["text"] for record in batch]
        ids = [record["record_id"] for record in batch]
        metadatas = [build_metadata(record) for record in batch]

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        print(
            f"Processed {min(start + len(batch), len(records))}"
            f" / {len(records)} records"
        )

    print("\n========== 5. VERIFY DATABASE ==========")

    total = collection.count()
    print("Expected records:", len(records))
    print("Stored records:", total)

    if total != len(records):
        raise RuntimeError(
            f"Record count mismatch: expected {len(records)}, got {total}"
        )

    sample = collection.get(
        limit=1,
        include=["documents", "metadatas"],
    )

    print("Sample ID:", sample["ids"][0])
    print("Sample text:", sample["documents"][0])
    print("Sample metadata:", sample["metadatas"][0])

    print("\n========== RESULT ==========")
    print("VideoRAG vector store: PASS")


if __name__ == "__main__":
    main()
