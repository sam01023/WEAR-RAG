"""
WEAR-RAG — Reranker
====================
Re-scores retrieved chunks using a cross-encoder that reads both the query
and the document text jointly, producing a fine-grained relevance score.

Why rerank?
    Bi-encoder retrieval (FAISS) is fast but treats query and document
    independently. Cross-encoders are slower but more accurate because
    they attend to the full (query, document) pair simultaneously.
    Reranking bridges the speed-accuracy tradeoff:
        FAISS retrieves the top-20 quickly → cross-encoder re-scores and
        keeps only the top-5 accurately.

Model: BAAI/bge-reranker-base
    - Lightweight cross-encoder fine-tuned for passage reranking
    - Outputs a raw logit; we apply sigmoid to get a [0, 1] relevance score
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from config import RetrievalConfig
from vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RankedChunk:
    """A chunk that has passed through both vector retrieval and cross-encoder reranking."""
    chunk_id: str
    source_doc_id: str
    text: str
    similarity_score: float   # from FAISS / bi-encoder
    reranker_score: float     # from cross-encoder (sigmoid → [0, 1])
    rank: int                 # position after reranking (1 = most relevant)

    @property
    def source_chunk(self):
        """Back-reference to underlying DocumentChunk (set externally if needed)."""
        return getattr(self, "_source_chunk", None)


class Reranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-base.

    Usage:
        reranker = Reranker(config)
        ranked = reranker.rerank(query, retrieved_chunks)
        top_k  = ranked[:5]
    """

    def __init__(self, config: RetrievalConfig):
        self.config = config
        logger.info("Loading reranker model: %s", config.reranker_model)
        # Import lazily to avoid top-level dependency if only embeddings are needed
        from sentence_transformers import CrossEncoder
        self.cross_encoder = CrossEncoder(
            config.reranker_model,
            max_length=512,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int = None) -> List[RankedChunk]:
        """
        Re-score *candidates* with the cross-encoder and return top_k.

        Args:
            query:      The query (or sub-query) string.
            candidates: Chunks retrieved from the vector store.
            top_k:      How many to keep (default: config.top_k_rerank).

        Returns:
            List of RankedChunk sorted by reranker_score descending.
        """
        if top_k is None:
            top_k = self.config.top_k_rerank

        if not candidates:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, c.chunk.text) for c in candidates]

        # Cross-encoder returns raw logits; apply sigmoid for interpretable [0, 1] scores
        raw_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        reranker_scores = self._sigmoid(raw_scores)

        # Combine with original similarity scores and sort
        ranked: List[RankedChunk] = []
        for idx, (candidate, reranker_score) in enumerate(zip(candidates, reranker_scores)):
            rc = RankedChunk(
                chunk_id=candidate.chunk.chunk_id,
                source_doc_id=candidate.chunk.source_doc_id,
                text=candidate.chunk.text,
                similarity_score=candidate.similarity_score,
                reranker_score=float(reranker_score),
                rank=0,  # set after sorting
            )
            rc._source_chunk = candidate.chunk
            ranked.append(rc)

        ranked.sort(key=lambda x: x.reranker_score, reverse=True)

        # Assign ranks (1-indexed)
        for i, item in enumerate(ranked):
            item.rank = i + 1

        top = ranked[:top_k]
        logger.debug(
            "Reranked %d candidates → kept %d | scores: %s",
            len(candidates), len(top),
            [f"{c.reranker_score:.3f}" for c in top],
        )
        return top

    def rerank_multi_query(
        self,
        queries: List[str],
        candidates_per_query: List[List[RetrievedChunk]],
        top_k: int = None,
    ) -> List[RankedChunk]:
        """
        Rerank candidates from multiple sub-queries together.
        Deduplicates by chunk_id, keeping the highest reranker score for each.

        Args:
            queries:               Sub-queries from query decomposition.
            candidates_per_query:  Parallel list of candidate lists.
            top_k:                 Final number of chunks to return.

        Returns:
            Deduplicated, sorted list of RankedChunk.
        """
        if top_k is None:
            top_k = self.config.top_k_rerank

        # Rerank each sub-query independently
        all_ranked: List[RankedChunk] = []
        for query, candidates in zip(queries, candidates_per_query):
            reranked = self.rerank(query, candidates, top_k=len(candidates))
            all_ranked.extend(reranked)

        # Deduplicate: keep the highest reranker_score per chunk_id
        best: dict = {}
        for rc in all_ranked:
            if rc.chunk_id not in best or rc.reranker_score > best[rc.chunk_id].reranker_score:
                best[rc.chunk_id] = rc

        deduplicated = sorted(best.values(), key=lambda x: x.reranker_score, reverse=True)

        # Re-assign ranks after deduplication
        for i, item in enumerate(deduplicated):
            item.rank = i + 1

        return deduplicated[:top_k]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.array(x, dtype=np.float64)))
