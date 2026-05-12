"""
WEAR-RAG — Weighted Evidence Aggregation  (Core Contribution)
==============================================================
This module is the heart of WEAR-RAG.

Standard RAG hands ALL retrieved documents to the LLM with equal weight.
This is noisy: irrelevant or low-quality chunks dilute the signal.

WEAR-RAG instead computes a composite Evidence Score for every candidate chunk:

    EvidenceScore = w_sim  × similarity_score     (bi-encoder relevance)
                  + w_rank × reranker_score        (cross-encoder relevance)
                  + w_dens × density_score         (information richness)

Only chunks above a minimum threshold pass through to generation.
The chunks are also ordered and can be optionally truncated to a token budget.

Information Density Score
--------------------------
A fast proxy for content richness. Rewards chunks that are:
  - Longer (more content)
  - Rich in named entities / technical terms (uppercase ratio as proxy)
  - Not overly redundant (penalises very short or very repetitive chunks)
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import AggregationConfig
from reranker import RankedChunk

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    """A single piece of evidence ready for LLM consumption."""
    chunk_id: str
    source_doc_id: str
    text: str

    # Component scores
    similarity_score: float
    reranker_score: float
    density_score: float
    evidence_score: float      # composite weighted score

    # Position in final ranked list (1 = most important)
    evidence_rank: int = 0

    # Component weights used (for interpretability)
    weights: Dict[str, float] = field(default_factory=dict)

    def as_context_string(self, include_score: bool = False) -> str:
        """Format this chunk as a labelled context block for the LLM prompt."""
        header = f"[Evidence {self.evidence_rank} | Source: {self.source_doc_id}"
        if include_score:
            header += f" | Score: {self.evidence_score:.3f}"
        header += "]"
        return f"{header}\n{self.text}"


class WeightedEvidenceAggregator:
    """
    Aggregates reranked chunks into a final, scored evidence set.

    Steps:
        1. Compute information density for each chunk.
        2. Compute composite evidence score.
        3. Filter chunks below the score threshold.
        4. Sort by evidence score (descending).
        5. Optionally deduplicate semantically similar chunks.
        6. Trim to a token budget for the LLM context window.
    """

    def __init__(self, config: AggregationConfig):
        self.config = config
        self._validate_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        ranked_chunks: List[RankedChunk],
        max_evidence: Optional[int] = None,
        token_budget: Optional[int] = None,
    ) -> List[EvidenceItem]:
        """
        Produce a filtered, scored, ordered list of evidence items.

        Args:
            ranked_chunks: Output from Reranker.rerank() or rerank_multi_query().
            max_evidence:  Hard cap on number of evidence items returned.
            token_budget:  Approximate token limit for total context (words × 1.3).

        Returns:
            List of EvidenceItem, sorted by evidence_score descending.
        """
        if not ranked_chunks:
            return []

        # Compute density scores
        density_scores = [self._information_density(rc.text) for rc in ranked_chunks]

        # Compute composite evidence scores
        items: List[EvidenceItem] = []
        for rc, density in zip(ranked_chunks, density_scores):
            score = self._evidence_score(rc.similarity_score, rc.reranker_score, density)

            if score < self.config.score_threshold:
                logger.debug("Filtered low-score chunk %s (score=%.3f)", rc.chunk_id, score)
                continue

            items.append(EvidenceItem(
                chunk_id=rc.chunk_id,
                source_doc_id=rc.source_doc_id,
                text=rc.text,
                similarity_score=rc.similarity_score,
                reranker_score=rc.reranker_score,
                density_score=density,
                evidence_score=score,
                weights={
                    "similarity": self.config.similarity_weight,
                    "reranker": self.config.reranker_weight,
                    "density": self.config.density_weight,
                },
            ))

        # Sort by composite score
        items.sort(key=lambda x: x.evidence_score, reverse=True)

        # Apply optional hard cap
        if max_evidence is not None:
            items = items[:max_evidence]

        # Trim to token budget
        if token_budget is not None:
            items = self._trim_to_budget(items, token_budget)

        # Assign evidence ranks
        for i, item in enumerate(items):
            item.evidence_rank = i + 1

        logger.info(
            "Aggregated %d chunks → %d evidence items (threshold=%.2f)",
            len(ranked_chunks), len(items), self.config.score_threshold,
        )
        return items

    def build_context(
        self,
        evidence_items: List[EvidenceItem],
        include_scores: bool = False,
    ) -> str:
        """
        Concatenate evidence items into a single context string for the LLM.

        Args:
            evidence_items: Output of aggregate().
            include_scores: If True, embed score annotations for interpretability.

        Returns:
            A multi-paragraph string with labelled evidence blocks.
        """
        if not evidence_items:
            return ""
        blocks = [item.as_context_string(include_score=include_scores) for item in evidence_items]
        return "\n\n".join(blocks)

    def score_summary(self, evidence_items: List[EvidenceItem]) -> str:
        """Return a human-readable summary table of evidence scores."""
        if not evidence_items:
            return "No evidence items."
        lines = [
            f"{'Rank':<5} {'Score':<8} {'Sim':<8} {'Rerank':<8} {'Density':<8} Source",
            "-" * 70,
        ]
        for item in evidence_items:
            snippet = item.text[:50].replace("\n", " ") + "…"
            lines.append(
                f"{item.evidence_rank:<5} {item.evidence_score:<8.3f} "
                f"{item.similarity_score:<8.3f} {item.reranker_score:<8.3f} "
                f"{item.density_score:<8.3f} [{item.source_doc_id}] {snippet}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Scoring internals
    # ------------------------------------------------------------------

    def _evidence_score(
        self, similarity: float, reranker: float, density: float
    ) -> float:
        """
        Composite evidence score.

            score = w_sim × sim + w_rank × reranker + w_dens × density
        """
        return (
            self.config.similarity_weight * similarity
            + self.config.reranker_weight * reranker
            + self.config.density_weight * density
        )

    @staticmethod
    def _information_density(text: str) -> float:
        """
        Heuristic measure of information density ∈ [0, 1].

        Combines:
            - Normalised length (prefers longer chunks with more content)
            - Type-token ratio (vocabulary diversity)
            - Named entity / technical term proxy (uppercase ratio)
        """
        words = text.split()
        if len(words) < 3:
            return 0.0

        # 1. Length factor: log-scaled, saturates around 200 words
        length_score = min(1.0, math.log1p(len(words)) / math.log1p(200))

        # 2. Type-token ratio (vocabulary richness), capped at 1.0
        unique_ratio = len(set(w.lower() for w in words)) / len(words)

        # 3. Named entity / technical term proxy: fraction of tokens that look
        #    like proper nouns or acronyms (first-char uppercase, not sentence-start)
        uppercase_words = sum(
            1 for i, w in enumerate(words) if i > 0 and len(w) > 1 and w[0].isupper()
        )
        entity_density = min(1.0, uppercase_words / max(1, len(words) - 1))

        # Weighted combination
        density = 0.5 * length_score + 0.3 * unique_ratio + 0.2 * entity_density
        return round(min(1.0, density), 4)

    # ------------------------------------------------------------------
    # Budget management
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_to_budget(items: List[EvidenceItem], token_budget: int) -> List[EvidenceItem]:
        """
        Keep the highest-scoring evidence items that fit within *token_budget*.
        Uses a rough words-to-tokens conversion of ×1.3.
        """
        kept: List[EvidenceItem] = []
        used_tokens = 0
        for item in items:
            # Rough token estimate: 1 word ≈ 1.3 tokens
            item_tokens = int(len(item.text.split()) * 1.3)
            if used_tokens + item_tokens > token_budget:
                logger.debug(
                    "Token budget exceeded at rank %d (%d/%d tokens).",
                    item.evidence_rank, used_tokens, token_budget,
                )
                break
            kept.append(item)
            used_tokens += item_tokens
        return kept

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_weights(self):
        total = (
            self.config.similarity_weight
            + self.config.reranker_weight
            + self.config.density_weight
        )
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"AggregationConfig weights must sum to 1.0, got {total:.4f}. "
                f"Adjust similarity_weight, reranker_weight, density_weight."
            )
