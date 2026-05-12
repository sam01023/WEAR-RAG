"""
WEAR-RAG — Embeddings
======================
Creates dense vector representations for document chunks and queries.

Model: BAAI/bge-small-en
  - Small, fast, English-only
  - Optimized for retrieval tasks
  - Outputs 384-dimensional vectors

BGE models expect a query prefix: "Represent this sentence: " for passages,
and "query: " prefix is recommended for queries.
"""

import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EmbeddingConfig
from document_processor import DocumentChunk

logger = logging.getLogger(__name__)

# BGE models perform best with these instruction prefixes
QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "
PASSAGE_INSTRUCTION = "Represent this passage for retrieval: "


class EmbeddingEngine:
    """
    Wraps SentenceTransformer to produce embeddings for passages and queries.

    All vectors are L2-normalized so that inner-product = cosine similarity,
    which is required for correct FAISS IndexFlatIP lookups.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        logger.info("Loading embedding model: %s", config.model_name)
        self.model = SentenceTransformer(config.model_name, device=config.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d", self.dimension)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """
        Embed a list of DocumentChunks.

        Returns:
            np.ndarray of shape (n_chunks, dim), dtype float32, L2-normalized.
        """
        texts = [PASSAGE_INSTRUCTION + chunk.text for chunk in chunks]
        return self._encode(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns:
            np.ndarray of shape (1, dim), dtype float32, L2-normalized.
        """
        text = QUERY_INSTRUCTION + query
        return self._encode([text])

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Embed multiple query strings.

        Returns:
            np.ndarray of shape (n_queries, dim), dtype float32, L2-normalized.
        """
        texts = [QUERY_INSTRUCTION + q for q in queries]
        return self._encode(texts)

    @property
    def underlying_model(self) -> SentenceTransformer:
        """Expose the raw model so other components (e.g. SemanticChunker) can reuse it."""
        return self.model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts in batches and return L2-normalized float32 vectors.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,   # L2 normalize → cosine via dot product
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two 1-D vectors.
        Works correctly even without pre-normalization.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
