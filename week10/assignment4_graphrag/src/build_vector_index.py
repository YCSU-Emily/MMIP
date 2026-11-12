import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"

CHUNKS_FILE = RESULTS / "paper_chunks.json"
DB_DIR = RESULTS / "chroma_vector_db"

MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
    / "5617a9f61b028005a4858fdac845db406aefb181"
)

COLLECTION_NAME = "graphrag_paper_chunks"
BATCH_SIZE = 8


def main():
    if DB_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing index: {DB_DIR}"
        )

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(CHUNKS_FILE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    chunks = data["chunks"]

    if not chunks:
        raise ValueError("Chunk dataset is empty.")

    print("=" * 65)
    print("BUILDING BGE-M3 VECTOR INDEX")
    print("=" * 65)
    print("Chunks:", len(chunks))
    print("Model:", MODEL_PATH)
    print("Device: CUDA")

    print("\n===== 1. LOAD EMBEDDING MODEL =====")

    model = SentenceTransformer(
        str(MODEL_PATH),
        local_files_only=True,
        device="cuda",
    )

    print("Embedding dimension:", model.get_sentence_embedding_dimension())

    print("\n===== 2. GENERATE EMBEDDINGS =====")

    texts = [item["text"] for item in chunks]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("Embedding shape:", embeddings.shape)

    if embeddings.shape != (len(chunks), 1024):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    print("\n===== 3. CREATE CHROMADB COLLECTION =====")

    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dimension": 1024,
            "source": "From Local to Global: A GraphRAG Approach",
        },
    )

    print("\n===== 4. INSERT CHUNKS =====")

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        collection.add(
            ids=[item["chunk_id"] for item in batch],
            documents=[item["text"] for item in batch],
            embeddings=embeddings[start:start + len(batch)].tolist(),
            metadatas=[
                {
                    "page": item["page"],
                    "chunk_index": item["chunk_index"],
                    "char_count": item["char_count"],
                }
                for item in batch
            ],
        )

        print(
            f"Inserted {min(start + len(batch), len(chunks))}"
            f"/{len(chunks)}"
        )

    print("\n===== 5. VERIFY INDEX =====")

    count = collection.count()
    stored = collection.get(
        limit=len(chunks),
        include=["documents", "metadatas"],
    )

    print("Collection:", collection.name)
    print("Stored vectors:", count)
    print("Stored documents:", len(stored["documents"]))
    print("Stored metadata:", len(stored["metadatas"]))

    if count != len(chunks):
        raise RuntimeError("Vector count mismatch.")

    if len(set(stored["ids"])) != len(chunks):
        raise RuntimeError("Duplicate or missing vector IDs.")

    print("\n===== BUILD COMPLETE =====")
    print("PASS: All chunks indexed successfully.")
    print("Database:", DB_DIR)


if __name__ == "__main__":
    main()
