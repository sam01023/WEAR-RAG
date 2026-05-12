"""
WEAR-RAG — Vector Store
========================
Stores chunk embeddings in a FAISS index and supports fast nearest-neighbor
retrieval. Also manages the mapping from FAISS integer IDs back to
DocumentChunk objects.

Index type: IndexFlatIP (exact inner-product / cosine search on L2-normalized vectors)
For very large corpora, swap to IndexIVFFlat or HNSW for speed.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from document_processor import DocumentChunk
from embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


class RetrievedChunk:
    """A chunk returned by the vector store, enriched with its similarity score."""

    def __init__(self, chunk: DocumentChunk, similarity_score: float, faiss_id: int):
        self.chunk = chunk
        self.similarity_score = similarity_score  # cosine similarity ∈ [0, 1]
        self.faiss_id = faiss_id

    def __repr__(self):
        return (
            f"RetrievedChunk(id={self.chunk.chunk_id!r}, "
            f"score={self.similarity_score:.4f})"
        )


class VectorStore:
    """
    FAISS-backed vector store for document chunk embeddings.

    Responsibilities:
        - Index chunks during ingestion.
        - Retrieve top-k most similar chunks for a given query embedding.
        - Persist / load index to / from disk.
    """

    def __init__(self, embedding_engine: EmbeddingEngine, store_path: str = "./faiss_index"):
        self.embedding_engine = embedding_engine
        self.store_path = store_path
        self.dimension = embedding_engine.dimension

        # FAISS index: Inner Product on L2-normalized vectors = cosine similarity
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)

        # Map from FAISS sequential integer ID → DocumentChunk
        self._id_to_chunk: Dict[int, DocumentChunk] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Embed and index a list of DocumentChunks.

        Args:
            chunks: Chunks to add. Can be called multiple times incrementally.
        """
        if not chunks:
            return

        logger.info("Embedding %d chunks...", len(chunks))
        embeddings = self.embedding_engine.embed_chunks(chunks)  # (n, dim) float32

        # Register each chunk in the ID map
        for chunk in chunks:
            self._id_to_chunk[self._next_id] = chunk
            self._next_id += 1

        self.index.add(embeddings)
        logger.info("Index size: %d vectors", self.index.ntotal)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> List[RetrievedChunk]:
        """
        Retrieve the top_k most similar chunks for *query*.

        Args:
            query: Raw query string (will be embedded internally).
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk, sorted by descending similarity score.
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty. Did you call add_chunks()?")
            return []

        query_embedding = self.embedding_engine.embed_query(query)  # (1, dim)
        effective_k = min(top_k, self.index.ntotal)

        # FAISS returns distances (inner product scores) and integer IDs
        scores, ids = self.index.search(query_embedding, effective_k)
        scores, ids = scores[0], ids[0]  # unwrap batch dimension

        results: List[RetrievedChunk] = []
        for score, fid in zip(scores, ids):
            if fid == -1:   # FAISS returns -1 for padded results
                continue
            chunk = self._id_to_chunk.get(int(fid))
            if chunk is not None:
                results.append(RetrievedChunk(chunk=chunk, similarity_score=float(score), faiss_id=int(fid)))

        return results

    def search_multi(self, queries: List[str], top_k: int = 20) -> List[List[RetrievedChunk]]:
        """
        Batch search for multiple queries (e.g., decomposed sub-questions).

        Returns:
            List of result lists, one per query.
        """
        return [self.search(q, top_k=top_k) for q in queries]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the FAISS index and chunk map to disk."""
        os.makedirs(self.store_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.store_path, "index.faiss"))
        with open(os.path.join(self.store_path, "chunks.pkl"), "wb") as f:
            pickle.dump({"id_to_chunk": self._id_to_chunk, "next_id": self._next_id}, f)
        logger.info("Vector store saved to %s", self.store_path)

    def load(self) -> bool:
        """
        Load a previously saved FAISS index from disk.

        Returns:
            True if loaded successfully, False if no saved index exists.
        """
        index_path = os.path.join(self.store_path, "index.faiss")
        chunks_path = os.path.join(self.store_path, "chunks.pkl")

        if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
            return False

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
        self._id_to_chunk = data["id_to_chunk"]
        self._next_id = data["next_id"]
        logger.info("Vector store loaded from %s (%d vectors)", self.store_path, self.index.ntotal)
        return True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Reverse-lookup a chunk by its string ID."""
        for chunk in self._id_to_chunk.values():
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
