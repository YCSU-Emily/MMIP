"""
Persistent ChromaDB vector store for Smart3C RAG.
"""

from pathlib import Path
from typing import Any, Sequence

import chromadb
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "vector_db"
DEFAULT_COLLECTION_NAME = "smart3c_faq"


class VectorStore:
    """Manage FAQ documents and embeddings in persistent ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")

        self.persist_directory = Path(persist_directory).expanduser()
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.collection_name = collection_name

    def add_documents(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: np.ndarray,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> int:
        """Add FAQ documents and their precomputed embeddings."""
        if isinstance(ids, str) or isinstance(documents, str):
            raise TypeError("ids and documents must be sequences.")

        ids = list(ids)
        documents = list(documents)
        vectors = np.asarray(embeddings, dtype=np.float32)

        if not ids:
            raise ValueError("At least one document is required.")

        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique.")

        if len(ids) != len(documents):
            raise ValueError("IDs and documents must have equal lengths.")

        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError("Embedding count must match document count.")

        if vectors.shape[1] != 1024:
            raise ValueError(
                f"Expected 1024-dimensional embeddings, got {vectors.shape[1]}."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain NaN or Inf values.")

        if any(not isinstance(item, str) or not item.strip()
               for item in ids + documents):
            raise ValueError("IDs and documents must be non-empty strings.")

        if metadatas is not None:
            metadatas = list(metadatas)
            if len(metadatas) != len(ids):
                raise ValueError("Metadata count must match document count.")

            for metadata in metadatas:
                if not isinstance(metadata, dict):
                    raise TypeError("Each metadata item must be a dictionary.")
                if not metadata:
                    raise ValueError("Metadata dictionaries must not be empty.")

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors.tolist(),
            metadatas=metadatas,
        )

        return len(ids)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the most similar FAQ documents."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        vector = np.asarray(query_embedding, dtype=np.float32)

        if vector.shape != (1024,):
            raise ValueError(
                f"Expected query shape (1024,), got {vector.shape}."
            )

        if not np.isfinite(vector).all():
            raise ValueError("Query embedding contains NaN or Inf values.")

        count = self.collection.count()
        if count == 0:
            return []

        result = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        matches = []

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        for item_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            matches.append({
                "id": item_id,
                "document": document,
                "metadata": metadata or {},
                "distance": float(distance),
                "similarity": 1.0 - float(distance),
            })

        return matches

    def count(self) -> int:
        """Return the number of stored documents."""
        return self.collection.count()
