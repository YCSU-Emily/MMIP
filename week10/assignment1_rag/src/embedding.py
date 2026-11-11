"""
BGE-M3 embedding module for Smart3C RAG.

Provides document and query embedding functions.
"""

from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DIM = 1024


class TextEmbedder:
    """Encode Chinese and multilingual text with BGE-M3."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 16,
        normalize_embeddings: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.model_name = model_name
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        if self.embedding_dim != DEFAULT_EMBEDDING_DIM:
            raise ValueError(
                f"Expected {DEFAULT_EMBEDDING_DIM} dimensions, "
                f"got {self.embedding_dim}."
            )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Convert a sequence of texts into a 2D embedding array."""
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not one string.")

        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every text must be a non-empty string.")

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.shape != (len(texts), self.embedding_dim):
            raise RuntimeError(
                f"Unexpected embedding shape: {embeddings.shape}"
            )

        if not np.isfinite(embeddings).all():
            raise RuntimeError("Embedding contains NaN or Inf values.")

        return embeddings

    def embed_documents(self, documents: Sequence[str]) -> np.ndarray:
        """Embed Knowledge Base documents."""
        return self._encode(documents)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one user query and return a 1D vector."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        return self._encode([query])[0]

    def get_embedding_dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self.embedding_dim
