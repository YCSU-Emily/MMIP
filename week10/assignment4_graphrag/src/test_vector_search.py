from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT / "results/chroma_vector_db"

MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
    / "5617a9f61b028005a4858fdac845db406aefb181"
)

QUERIES = [
    "How does GraphRAG construct a knowledge graph from source documents?",
    "How does GraphRAG use community summaries to answer global questions?",
    "Which evaluation metrics are used to compare different RAG approaches?",
]


def main():
    print("=" * 65)
    print("VECTOR RAG RETRIEVAL SMOKE TEST")
    print("=" * 65)

    print("\n===== 1. LOAD MODEL AND DATABASE =====")

    model = SentenceTransformer(
        str(MODEL_PATH),
        local_files_only=True,
        device="cuda",
    )

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection("graphrag_paper_chunks")

    print("Collection count:", collection.count())

    if collection.count() != 95:
        raise RuntimeError("Expected 95 indexed chunks.")

    print("\n===== 2. RUN TOP-5 RETRIEVAL =====")

    for index, query in enumerate(QUERIES, 1):
        print(f"\n{'=' * 65}")
        print(f"QUERY {index}: {query}")
        print("=" * 65)

        query_vector = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        result = collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            1,
        ):
            print(f"\n[{rank}] {chunk_id}")
            print("Page:", metadata["page"])
            print("Cosine distance:", round(distance, 4))
            print("Text:", document[:350].replace("\n", " "))

        print("\nRetrieved:", len(ids), "chunks")

    print("\n===== TEST COMPLETE =====")
    print("PASS: All three queries retrieved five chunks.")


if __name__ == "__main__":
    main()
