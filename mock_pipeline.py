"""
Lightweight mock pipeline for the web demo.

This avoids loading heavy ML models and provides a deterministic,
explainable demo response based on simple keyword overlap scoring.
"""
import re
from typing import List, Tuple, Dict


demo_documents = [
    {"id": "transformers_overview", "text": (
        "Transformers are deep learning models that use self-attention mechanisms. "
        "Self-attention allows the model to weigh the importance of each token relative "
        "to every other token in the sequence. This enables highly parallelised computation "
        "compared to sequential RNN processing.")},
    {"id": "rnn_limitations", "text": (
        "Recurrent Neural Networks process sequences one token at a time. "
        "This sequential nature prevents parallelisation and leads to vanishing gradient "
        "problems on long sequences. LSTMs and GRUs partially address these issues but "
        "still struggle with very long-range dependencies.")},
    {"id": "attention_mechanism", "text": (
        "The attention mechanism computes a weighted sum of value vectors, where weights "
        "are derived from the compatibility between query and key vectors. Multi-head "
        "attention applies this operation in parallel across multiple subspaces.")},
    {"id": "gpu_acceleration", "text": (
        "Transformer architectures are highly amenable to GPU acceleration because their "
        "matrix operations can be batched across the full sequence length. RNNs require "
        "sequential computation, making GPU utilisation far lower.")},
    {"id": "bert_gpt", "text": (
        "BERT and GPT are prominent transformer-based models. BERT is used for "
        "understanding tasks while GPT excels at text generation. Both use self-attention "
        "as their core mechanism.")},
]


_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = set([
    "the", "is", "a", "an", "and", "or", "to", "of", "in", "for", "on", "that", "this",
    "are", "be", "as", "with", "it", "its", "by", "from"
])


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _score(question: str, text: str) -> float:
    qtokens = set(_tokenize(question))
    if not qtokens:
        return 0.0
    dtokens = set(_tokenize(text))
    common = qtokens & dtokens
    return float(len(common)) / float(len(qtokens))


def _best_sentences(question: str, text: str, max_sentences: int = 2) -> str:
    sents = re.split(r'(?<=[.!?])\s+', text)
    selected = []
    qtokens = set(_tokenize(question))
    for s in sents:
        if len(selected) >= max_sentences:
            break
        if qtokens & set(_tokenize(s)):
            selected.append(s.strip())
    if not selected and sents:
        # fallback: use the first sentence(s)
        return " ".join(sents[:max_sentences]).strip()
    return " ".join(selected)


class MockPipeline:
    """Simple, dependency-free pipeline used by the web demo."""

    def __init__(self, documents: List[dict] = None):
        self.docs = documents or demo_documents

    def answer(self, question: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        """
        Return a tuple (answer_text, evidence_list).

        `evidence_list` is a list of dicts with keys: `source_doc_id`, `text`, `score`.
        """
        scored = []
        for d in self.docs:
            score = _score(question, d["text"]) if question.strip() else 0.0
            scored.append({"id": d["id"], "text": d["text"], "score": score})

        # If all scores zero (e.g., very different phrasing), fall back to fuzzy substring match
        if all(item["score"] == 0.0 for item in scored):
            # cheap heuristic: use sequence inclusion / substring matches
            q = question.lower().strip()
            for item in scored:
                if q and q in item["text"].lower():
                    item["score"] = 0.5

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = [s for s in scored if s["score"] > 0.0][:top_k]
        if not top:
            top = scored[:top_k]

        # Build a lightweight answer by extracting the most relevant sentences
        parts = []
        for item in top:
            parts.append(_best_sentences(question, item["text"]))
        answer = " ".join(parts).strip() or "I'm unable to find relevant evidence for that question."

        evidence = []
        for item in top:
            evidence.append({
                "source_doc_id": item["id"],
                "text": item["text"],
                "score": round(float(item["score"]), 3),
            })

        return answer, evidence


if __name__ == "__main__":
    p = MockPipeline()
    q = "Why are transformers better than RNNs?"
    a, e = p.answer(q)
    print("Question:", q)
    print("Answer:", a)
    print("Evidence:")
    for it in e:
        print(it)
