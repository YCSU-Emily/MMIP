"""
FAQ retrieval module for Smart3C RAG.
Uses the Group Split V3 vector database.
"""

from pathlib import Path
from typing import Any

from .embedding import TextEmbedder
from .vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "vector_db_grouped"
DEFAULT_COLLECTION_NAME = "smart3c_faq_grouped"


class FAQRetriever:
    """Retrieve relevant FAQ answers and their source references."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        device: str | None = None,
    ) -> None:
        db_path = Path(persist_directory).expanduser()

        if not db_path.exists():
            raise FileNotFoundError(
                f"Vector database does not exist: {db_path}"
            )

        self.embedder = TextEmbedder(device=device)
        self.vector_store = VectorStore(
            persist_directory=db_path,
            collection_name=collection_name,
        )

        if self.vector_store.count() == 0:
            raise RuntimeError("The FAQ collection is empty.")

    @staticmethod
    def _parse_document(document: str) -> dict[str, str]:
        """Extract question and answer from the stored FAQ document."""
        fields: dict[str, str] = {}

        for line in document.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                key = key.strip()
                if key in {"Question", "Answer", "Keywords", "Category"}:
                    fields[key] = value.strip()

        return fields

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k FAQ records for one user question."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_embedding = self.embedder.embed_query(query)
        matches = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        results = []

        for rank, match in enumerate(matches, start=1):
            metadata = match["metadata"]
            parsed = self._parse_document(match["document"])

            results.append({
                "rank": rank,
                "id": match["id"],
                "question": parsed.get("Question", ""),
                "answer": parsed.get("Answer", ""),
                "keywords": metadata.get(
                    "keywords", parsed.get("Keywords", "")
                ),
                "category": metadata.get(
                    "category", parsed.get("Category", "")
                ),
                "difficulty": metadata.get("difficulty", ""),
                "source_id": metadata.get("source_id", ""),
                "source_title": metadata.get("source_title", ""),
                "source_publisher": metadata.get("source_publisher", ""),
                "source_url": metadata.get("source_url", ""),
                "verified_at": metadata.get("verified_at", ""),
                "distance": match["distance"],
                "similarity": match["similarity"],
            })

        return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Search Smart3C FAQ using Group Split V3."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User question to search.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of FAQ results to retrieve.",
    )
    args = parser.parse_args()

    retriever = FAQRetriever()
    results = retriever.retrieve(args.query, top_k=args.top_k)

    print("===== SMART3C FAQ RETRIEVAL =====")
    print("Query:", args.query)
    print("Top-K:", args.top_k)
    print("Results:", len(results))

    for item in results:
        print(f"\n===== RANK {item['rank']} =====")
        print("ID:", item["id"])
        print("Question:", item["question"])
        print("Answer:", item["answer"])
        print("Category:", item["category"])
        print("Similarity:", f"{item['similarity']:.4f}")
        print("Distance:", f"{item['distance']:.4f}")
        print("Source:", item["source_title"])
        print("Publisher:", item["source_publisher"])
        print("URL:", item["source_url"])
        print("Verified at:", item["verified_at"])


if __name__ == "__main__":
    main()
