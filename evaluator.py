"""
WEAR-RAG — Evaluator (v2)
=========================
Metrics:
    • Exact Match (EM)
    • Token-level F1
    • ROUGE-L
    • BLEU-1
    • Mean Reciprocal Rank (MRR)
    • Retrieval Precision

Results are saved to CSV for easy reporting.
"""

import csv
import logging
import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    question_id: str
    question: str
    predicted_answer: str
    gold_answer: str
    retrieved_chunk_ids: List[str]
    gold_supporting_doc_ids: List[str]
    evidence_scores: List[float] = field(default_factory=list)

    exact_match: float = 0.0
    f1: float = 0.0
    rouge_l: float = 0.0
    bleu_1: float = 0.0
    mrr: float = 0.0
    retrieval_precision: float = 0.0


@dataclass
class EvaluationReport:
    system_name: str
    num_samples: int
    avg_exact_match: float
    avg_f1: float
    avg_rouge_l: float
    avg_bleu_1: float
    avg_mrr: float
    avg_retrieval_precision: float
    per_sample: List[PredictionResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  System  : {self.system_name}\n"
            f"  Samples : {self.num_samples}\n"
            f"  EM      : {self.avg_exact_match:.4f}\n"
            f"  F1      : {self.avg_f1:.4f}\n"
            f"  ROUGE-L : {self.avg_rouge_l:.4f}\n"
            f"  BLEU-1  : {self.avg_bleu_1:.4f}\n"
            f"  MRR     : {self.avg_mrr:.4f}\n"
            f"  RetPrec : {self.avg_retrieval_precision:.4f}\n"
            f"{'='*55}"
        )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L based on Longest Common Subsequence."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    # LCS via dynamic programming
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == gold_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]

    precision = lcs / m if m > 0 else 0.0
    recall = lcs / n if n > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu_1(prediction: str, gold: str) -> float:
    """Unigram BLEU with brevity penalty."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens:
        return 0.0

    gold_counter = Counter(gold_tokens)
    matches = sum(min(count, gold_counter[token]) for token, count in Counter(pred_tokens).items())
    precision = matches / len(pred_tokens)

    # Brevity penalty
    bp = 1.0 if len(pred_tokens) >= len(gold_tokens) else math.exp(1 - len(gold_tokens) / len(pred_tokens))
    return bp * precision


def mean_reciprocal_rank(retrieved_ids: List[str], gold_doc_ids: List[str]) -> float:
    """
    MRR: reciprocal of the rank of the first relevant document.
    Rewards systems that put relevant docs near the top.
    """
    gold_set = set(gold_doc_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def retrieval_precision(retrieved_ids: List[str], gold_doc_ids: List[str]) -> float:
    if not retrieved_ids:
        return 0.0
    gold_set = set(gold_doc_ids)
    hits = sum(1 for doc_id in retrieved_ids if doc_id in gold_set)
    return hits / len(retrieved_ids)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:

    def load_hotpot_samples(self, n: int = 100, split: str = "validation") -> List[dict]:
        """Load HotpotQA samples. Use n=0 to load ALL samples from the split."""
        try:
            from datasets import load_dataset
            load_all = (n == 0)
            logger.info("Loading HotpotQA %s split (%s)...", split,
                         "ALL" if load_all else f"n={n}")
            ds = load_dataset("hotpot_qa", "distractor", split=split)
            samples = []
            for i, item in enumerate(ds):
                if not load_all and i >= n:
                    break
                documents = []
                for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
                    documents.append({"id": title, "text": " ".join(sentences)})
                samples.append({
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "supporting_titles": item["supporting_facts"]["title"],
                    "documents": documents,
                })
            logger.info("Loaded %d HotpotQA samples.", len(samples))
            return samples
        except Exception as exc:
            logger.error("Failed to load HotpotQA: %s", exc)
            return []

    def evaluate(
        self,
        pipeline_fn: Callable[[dict], Tuple[str, List[str], List[float]]],
        samples: List[dict],
        system_name: str = "System",
        save_csv: Optional[str] = None,
    ) -> EvaluationReport:

        results: List[PredictionResult] = []
        totals = dict(em=0.0, f1=0.0, rl=0.0, b1=0.0, mrr=0.0, rp=0.0)

        for idx, sample in enumerate(samples):
            logger.info("[%d/%d] %s: %s", idx+1, len(samples), system_name, sample["question"][:60])

            try:
                predicted, retrieved_ids, scores = pipeline_fn(sample)
            except Exception as exc:
                logger.warning("Pipeline error on %s: %s", sample["id"], exc)
                predicted, retrieved_ids, scores = "", [], []

            em  = exact_match(predicted, sample["answer"])
            f1  = token_f1(predicted, sample["answer"])
            rl  = rouge_l(predicted, sample["answer"])
            b1  = bleu_1(predicted, sample["answer"])
            mrr_score = mean_reciprocal_rank(retrieved_ids, sample["supporting_titles"])
            rp  = retrieval_precision(retrieved_ids, sample["supporting_titles"])

            result = PredictionResult(
                question_id=sample["id"],
                question=sample["question"],
                predicted_answer=predicted,
                gold_answer=sample["answer"],
                retrieved_chunk_ids=retrieved_ids,
                gold_supporting_doc_ids=sample["supporting_titles"],
                evidence_scores=scores,
                exact_match=em, f1=f1, rouge_l=rl, bleu_1=b1, mrr=mrr_score,
                retrieval_precision=rp,
            )
            results.append(result)
            totals["em"] += em; totals["f1"] += f1; totals["rl"] += rl
            totals["b1"] += b1; totals["mrr"] += mrr_score; totals["rp"] += rp

        n = max(len(results), 1)
        report = EvaluationReport(
            system_name=system_name,
            num_samples=len(results),
            avg_exact_match=totals["em"]/n,
            avg_f1=totals["f1"]/n,
            avg_rouge_l=totals["rl"]/n,
            avg_bleu_1=totals["b1"]/n,
            avg_mrr=totals["mrr"]/n,
            avg_retrieval_precision=totals["rp"]/n,
            per_sample=results,
        )
        logger.info(report.summary())

        if save_csv:
            self._save_csv(results, system_name, save_csv)

        return report

    def _save_csv(self, results: List[PredictionResult], system_name: str, path: str):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "system", "question_id", "question", "gold_answer", "predicted_answer",
                "exact_match", "f1", "rouge_l", "bleu_1", "mrr", "retrieval_precision"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "system": system_name,
                    "question_id": r.question_id,
                    "question": r.question,
                    "gold_answer": r.gold_answer,
                    "predicted_answer": r.predicted_answer,
                    "exact_match": round(r.exact_match, 4),
                    "f1": round(r.f1, 4),
                    "rouge_l": round(r.rouge_l, 4),
                    "bleu_1": round(r.bleu_1, 4),
                    "mrr": round(r.mrr, 4),
                    "retrieval_precision": round(r.retrieval_precision, 4),
                })
        logger.info("Results saved to %s", path)

    @staticmethod
    def compare_reports(reports: List[EvaluationReport]) -> str:
        header = f"\n{'System':<28} {'EM':>7} {'F1':>7} {'ROUGE-L':>9} {'BLEU-1':>8} {'MRR':>7} {'RetPrec':>9}"
        sep = "-" * 80
        rows = [header, sep]
        for r in reports:
            rows.append(
                f"{r.system_name:<28} "
                f"{r.avg_exact_match:>7.4f} "
                f"{r.avg_f1:>7.4f} "
                f"{r.avg_rouge_l:>9.4f} "
                f"{r.avg_bleu_1:>8.4f} "
                f"{r.avg_mrr:>7.4f} "
                f"{r.avg_retrieval_precision:>9.4f}"
            )
        rows.append(sep)
        return "\n".join(rows)
