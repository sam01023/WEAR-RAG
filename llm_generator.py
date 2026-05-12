"""
WEAR-RAG — LLM Generator
=========================
Sends the assembled evidence context + question to a local Ollama LLM
and returns the generated answer.

Model: Mistral 7B (via Ollama)

Prompt design:
    - System: Strict instructions to answer from evidence only.
    - Context: Labelled evidence blocks from WeightedEvidenceAggregator.
    - User:    The original question.

Includes a MockGenerator for offline testing / unit tests without Ollama.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from evidence_aggregator import EvidenceItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the question using ONLY the evidence provided below.
Be concise and factual. If the evidence does not contain the answer, say "I don't know."
Do NOT fabricate information."""

USER_TEMPLATE = """{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, evidence_items: List[EvidenceItem]) -> str:
        """Generate an answer given the question and evidence."""


# ---------------------------------------------------------------------------
# Ollama Generator
# ---------------------------------------------------------------------------

class OllamaGenerator(BaseGenerator):
    """
    Generates answers using a locally running Ollama model.

    Prerequisites:
        1. Install Ollama:  https://ollama.com
        2. Pull a model:    ollama pull mistral
        3. Ollama must be running: ollama serve
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 512,
        include_scores_in_context: bool = False,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.include_scores_in_context = include_scores_in_context

    def generate(self, question: str, evidence_items: List[EvidenceItem]) -> str:
        """
        Call the Ollama chat API and return the generated answer.

        Args:
            question:       Original user question.
            evidence_items: Scored evidence from WeightedEvidenceAggregator.

        Returns:
            Generated answer string, or error message on failure.
        """
        context = self._build_context(evidence_items)
        if not context:
            return "No relevant evidence found to answer this question."

        user_message = USER_TEMPLATE.format(context=context, question=question)

        try:
            import requests
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                "stream": False,
            }
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            answer = response.json()["message"]["content"].strip()
            logger.debug("Generated answer (%d chars)", len(answer))
            return answer

        except Exception as exc:
            logger.error("OllamaGenerator failed: %s", exc)
            return f"[Generation error: {exc}]"

    def _build_context(self, evidence_items: List[EvidenceItem]) -> str:
        if not evidence_items:
            return ""
        blocks = [item.as_context_string(include_score=self.include_scores_in_context)
                  for item in evidence_items]
        return "\n\n".join(blocks)

    def health_check(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Mock Generator (for testing without Ollama)
# ---------------------------------------------------------------------------

class MockGenerator(BaseGenerator):
    """
    Returns a deterministic canned response.
    Useful for unit-testing the pipeline without a running LLM.
    """

    def generate(self, question: str, evidence_items: List[EvidenceItem]) -> str:
        if not evidence_items:
            return "No evidence available."
        best = evidence_items[0]
        return (
            f"[MOCK ANSWER] Based on the top-scored evidence (score={best.evidence_score:.3f}) "
            f"from source '{best.source_doc_id}': {best.text[:200]}…"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_generator(use_mock: bool = False, **kwargs) -> BaseGenerator:
    """
    Build the appropriate generator.

    Args:
        use_mock: If True, return MockGenerator (no Ollama needed).
    """
    if use_mock:
        return MockGenerator()
    return OllamaGenerator(**kwargs)
