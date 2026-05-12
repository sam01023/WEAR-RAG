"""
WEAR-RAG — Query Decomposer
============================
Breaks complex, multi-hop questions into simpler sub-questions so each
retrieval step can target a focused piece of knowledge.

Why decompose?
    "Why are transformers better than RNNs?" requires knowledge about
    both architectures. A single query embedding may not capture both
    aspects well. Decomposition retrieves complementary evidence.

Two strategies are implemented:
    1. RuleBasedDecomposer  — fast, no external dependency, handles common patterns.
    2. LLMDecomposer        — uses the local Ollama model for richer decomposition.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseDecomposer(ABC):
    @abstractmethod
    def decompose(self, query: str) -> List[str]:
        """
        Decompose *query* into a list of sub-queries.
        Always includes the original query as one of the results.
        """


# ---------------------------------------------------------------------------
# Rule-Based Decomposer
# ---------------------------------------------------------------------------

COMPARISON_TRIGGERS = re.compile(
    r'\b(better|worse|differ|compare|versus|vs\.?|advantage|disadvantage|'
    r'more|less|faster|slower|prefer|over)\b', re.IGNORECASE
)

CAUSAL_TRIGGERS = re.compile(
    r'\b(why|because|cause|reason|explain|how does|what makes)\b', re.IGNORECASE
)

MULTI_ENTITY_TRIGGERS = re.compile(
    r'\b(and|both|between|each|all|every)\b', re.IGNORECASE
)


class RuleBasedDecomposer(BaseDecomposer):
    """
    Decomposes queries using heuristic patterns.

    Patterns handled:
        - Comparison questions (A vs B)
        - Causal questions (why / how)
        - Multi-entity questions (A and B)
        - Compound questions joined by "and"
    """

    def __init__(self, max_sub_queries: int = 4):
        self.max_sub_queries = max_sub_queries

    def decompose(self, query: str) -> List[str]:
        sub_queries = [query]   # always include the original

        # --- Comparison: "Why is X better than Y?" ---
        comparison_match = re.search(
            r'(.+?)\s+(?:better|worse|faster|slower|differ|vs\.?|versus)\s+(.+?)[\?.]?$',
            query, re.IGNORECASE
        )
        if comparison_match:
            concept_a = comparison_match.group(1).strip()
            concept_b = comparison_match.group(2).strip()
            sub_queries += [
                f"What is {concept_a}?",
                f"What is {concept_b}?",
                f"What are the advantages of {concept_a}?",
                f"What are the limitations of {concept_b}?",
            ]

        # --- Causal: "Why does X happen?" ---
        elif CAUSAL_TRIGGERS.search(query):
            # Extract the main subject
            subject = re.sub(r'^(why|how|what makes?)\s+', '', query, flags=re.IGNORECASE).strip('?.')
            sub_queries += [
                f"What is {subject}?",
                f"How does {subject} work?",
            ]

        # --- Compound: "What is X and Y?" ---
        elif ' and ' in query.lower():
            parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
            if len(parts) == 2:
                sub_queries += [p.strip() + '?' for p in parts]

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in sub_queries:
            key = q.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(q)

        return unique[:self.max_sub_queries]


# ---------------------------------------------------------------------------
# LLM-Based Decomposer (uses Ollama)
# ---------------------------------------------------------------------------

class LLMDecomposer(BaseDecomposer):
    """
    Uses a local LLM (via Ollama) to intelligently decompose complex queries.

    Falls back to RuleBasedDecomposer if Ollama is unavailable.
    """

    SYSTEM_PROMPT = (
        "You are a question decomposition assistant. "
        "Given a complex question, output 2–4 simpler sub-questions that together "
        "cover all the information needed to answer the original question. "
        "Output ONLY the sub-questions, one per line, no numbering or extra text."
    )

    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434",
                 fallback: Optional[BaseDecomposer] = None):
        self.model = model
        self.base_url = base_url
        self._fallback = fallback or RuleBasedDecomposer()

    def decompose(self, query: str) -> List[str]:
        try:
            import requests
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                "stream": False,
            }
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]

            # Parse one sub-question per non-empty line
            sub_queries = [line.strip() for line in content.splitlines() if line.strip()]
            # Always prepend the original
            result = [query] + [q for q in sub_queries if q.lower() != query.lower()]
            logger.debug("LLM decomposed %r → %d sub-queries", query, len(result))
            return result

        except Exception as exc:
            logger.warning("LLMDecomposer failed (%s), using rule-based fallback.", exc)
            return self._fallback.decompose(query)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_decomposer(use_llm: bool = False, **kwargs) -> BaseDecomposer:
    """
    Build the appropriate decomposer.

    Args:
        use_llm: If True, use LLMDecomposer with rule-based fallback.
                 If False, use RuleBasedDecomposer.
    """
    if use_llm:
        return LLMDecomposer(**kwargs)
    return RuleBasedDecomposer()
