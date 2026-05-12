"""
WEAR-RAG — Comprehensive Full-Dataset Evaluation
==================================================
Runs ALL pipeline variants on the full HotpotQA validation split
with checkpoint / resume support.

Evaluated Systems:
    Core Pipeline Variants (7):
        1. Naive RAG (BM25)         — keyword retrieval, no embeddings
        2. Baseline RAG (k=5)       — dense retrieval, top-5
        3. Baseline RAG (k=10)      — dense retrieval, top-10
        4. Decomposition-Only       — query decomposition + diversity
        5. Rerank-Only              — reranking without decomposition
        6. Hybrid Retrieval         — BM25 + Dense (RRF fusion)
        7. WEAR-RAG (Full)          — full pipeline (decomp+rerank+aggregation)

    Aggregation Ablations (2):
        8. WEAR-RAG (avg-agg)       — average aggregation
        9. WEAR-RAG (max-agg)       — max aggregation

Usage:
    # Full dataset with Ollama
    python run_full_eval.py --full

    # Small test run (mock LLM)
    python run_full_eval.py --samples 10 --mock

    # Resume interrupted run
    python run_full_eval.py --full --resume

    # Specific systems only
    python run_full_eval.py --full --systems "Naive RAG (BM25),WEAR-RAG (Full)"
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# ── Suppress noisy third-party loggers ──────────────────────────────────────
for _noisy in ("httpx", "httpcore", "huggingface_hub", "huggingface_hub.utils._http",
               "sentence_transformers", "transformers", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wear_rag.full_eval")

# ── Project imports ─────────────────────────────────────────────────────────
from config import WEARRAGConfig, DEFAULT_CONFIG
from evaluator import Evaluator, EvaluationReport
from visualizer import MatplotlibVisualizer
from main import (
    ModelRegistry, NaiveRAG, BaselineRAG, ImprovedRAG,
    RerankOnlyRAG, HybridRAG, WEARRAG, VectorStore
)

# ── Constants ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "eval_results")
CHECKPOINT_DIR = os.path.join(ROOT, "eval_checkpoints")
CHECKPOINT_INTERVAL = 50  # save checkpoint every N samples


# ===========================================================================
# Checkpoint Manager
# ===========================================================================

class CheckpointManager:
    """Save and resume evaluation progress."""

    def __init__(self, system_name: str):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        safe_name = system_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        self.path = os.path.join(CHECKPOINT_DIR, f"ckpt_{safe_name}.json")

    def save(self, completed_ids: List[str], partial_results: List[dict]):
        data = {
            "completed_ids": completed_ids,
            "partial_results": partial_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Checkpoint saved: %d samples completed.", len(completed_ids))

    def load(self) -> Tuple[List[str], List[dict]]:
        if not os.path.exists(self.path):
            return [], []
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Checkpoint loaded: %d samples already completed.", len(data["completed_ids"]))
        return data["completed_ids"], data["partial_results"]

    def clear(self):
        if os.path.exists(self.path):
            os.remove(self.path)


# ===========================================================================
# System Builder
# ===========================================================================

def build_all_systems(registry: ModelRegistry, use_mock: bool = False) -> Dict[str, callable]:
    """Return a dict of system_name -> builder_function."""
    return {
        # --- Core Pipeline Variants ---
        "Naive RAG (BM25)":     lambda: NaiveRAG(registry, use_mock_llm=use_mock),
        "Baseline RAG (k=5)":   lambda: BaselineRAG(registry, use_mock_llm=use_mock, top_k=5),
        "Baseline RAG (k=10)":  lambda: BaselineRAG(registry, use_mock_llm=use_mock, top_k=10),
        "Decomposition-Only":   lambda: ImprovedRAG(registry, use_mock_llm=use_mock),
        "Rerank-Only":          lambda: RerankOnlyRAG(registry, use_mock_llm=use_mock),
        "Hybrid Retrieval":     lambda: HybridRAG(registry, use_mock_llm=use_mock),
        "WEAR-RAG (Full)":      lambda: WEARRAG(registry, use_mock_llm=use_mock, aggregation_mode="weighted"),

        # --- Aggregation Ablations ---
        "WEAR-RAG (avg-agg)":   lambda: WEARRAG(registry, use_mock_llm=use_mock, aggregation_mode="average"),
        "WEAR-RAG (max-agg)":   lambda: WEARRAG(registry, use_mock_llm=use_mock, aggregation_mode="max"),
    }


# ===========================================================================
# Evaluation Runner
# ===========================================================================

def evaluate_system(
    system_name: str,
    build_fn: callable,
    samples: List[dict],
    evaluator: Evaluator,
    resume: bool = False,
) -> EvaluationReport:
    """
    Run evaluation for a single system with checkpoint support.
    """
    ckpt = CheckpointManager(system_name)
    completed_ids, partial_results = ([], []) if not resume else ckpt.load()
    completed_set = set(completed_ids)

    # Filter out already-completed samples
    remaining = [s for s in samples if s["id"] not in completed_set]
    logger.info(
        "[%s] %d total, %d already done, %d remaining.",
        system_name, len(samples), len(completed_ids), len(remaining),
    )

    if not remaining and completed_ids:
        logger.info("[%s] All samples already completed (checkpoint). Skipping.", system_name)
        # Rebuild report from checkpoint
        return _build_report_from_checkpoint(system_name, partial_results)

    pipeline = build_fn()

    for idx, sample in enumerate(remaining):
        progress = len(completed_ids) + idx + 1
        total = len(samples)
        if progress % 10 == 0 or progress == total:
            logger.info("[%s] %d/%d (%.1f%%)", system_name, progress, total, progress / total * 100)

        try:
            # Fresh vector/BM25 store per sample
            if hasattr(pipeline, 'vector_store'):
                pipeline.vector_store = VectorStore(
                    pipeline.embedding_engine,
                    os.path.join("/tmp", f"{system_name.replace(' ', '_')}_{sample['id']}")
                )
            if hasattr(pipeline, '_chunks'):
                pipeline._chunks = []
                pipeline._bm25 = None

            pipeline.ingest(sample["documents"])
            predicted, retrieved_ids, scores = pipeline.answer(sample["question"])
        except Exception as exc:
            logger.warning("[%s] Error on %s: %s", system_name, sample["id"], exc)
            predicted, retrieved_ids, scores = "", [], []

        partial_results.append({
            "id": sample["id"],
            "predicted": predicted,
            "retrieved_ids": retrieved_ids,
            "scores": scores,
        })
        completed_ids.append(sample["id"])

        # Checkpoint
        if (len(completed_ids)) % CHECKPOINT_INTERVAL == 0:
            ckpt.save(completed_ids, partial_results)

    # Final checkpoint save
    ckpt.save(completed_ids, partial_results)

    # Build report using evaluator
    safe_name = system_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    csv_path = os.path.join(RESULTS_DIR, f"results_{safe_name}.csv")

    # Re-create the pipeline function that returns cached results
    results_by_id = {r["id"]: r for r in partial_results}

    def pipeline_fn(sample):
        r = results_by_id[sample["id"]]
        return r["predicted"], r["retrieved_ids"], r["scores"]

    report = evaluator.evaluate(
        pipeline_fn,
        samples,
        system_name=system_name,
        save_csv=csv_path,
    )

    # Clear checkpoint after successful evaluation
    ckpt.clear()
    return report


def _build_report_from_checkpoint(system_name: str, partial_results: List[dict]) -> EvaluationReport:
    """Utility to rebuild a report from checkpoint data (without re-evaluating)."""
    from evaluator import (
        PredictionResult, exact_match, token_f1, rouge_l, bleu_1,
        mean_reciprocal_rank, retrieval_precision,
    )
    # We don't have gold answers in checkpoint, so we can't recompute metrics here.
    # Instead, we return a placeholder report and let the main loop handle it.
    return EvaluationReport(
        system_name=system_name,
        num_samples=len(partial_results),
        avg_exact_match=0.0, avg_f1=0.0, avg_rouge_l=0.0,
        avg_bleu_1=0.0, avg_mrr=0.0, avg_retrieval_precision=0.0,
    )


# ===========================================================================
# Main Runner
# ===========================================================================

def run_full_evaluation(
    n_samples: int = 0,
    use_mock: bool = False,
    resume: bool = False,
    selected_systems: Optional[List[str]] = None,
):
    """
    Run the full evaluation pipeline.

    Args:
        n_samples:  0 = full dataset, otherwise limit to n.
        use_mock:   True to use MockGenerator (no Ollama needed).
        resume:     True to resume from checkpoints.
        selected_systems: List of system names to evaluate (None = all).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = DEFAULT_CONFIG
    evaluator = Evaluator()

    # ── Load HotpotQA ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("WEAR-RAG Comprehensive Evaluation")
    logger.info("=" * 60)

    samples = evaluator.load_hotpot_samples(n=n_samples)
    if not samples:
        logger.error("No samples loaded. Install `datasets`: pip install datasets")
        return
    logger.info("Loaded %d HotpotQA samples.", len(samples))

    # ── Load shared models ──────────────────────────────────────────────────
    logger.info("Loading shared models (embedding + reranker)...")
    registry = ModelRegistry(config)
    _ = registry.embedding_engine
    _ = registry.reranker
    logger.info("Models ready.")

    # ── Build system registry ───────────────────────────────────────────────
    all_systems = build_all_systems(registry, use_mock=use_mock)

    if selected_systems:
        systems_to_eval = {
            name: fn for name, fn in all_systems.items()
            if name in selected_systems
        }
        if not systems_to_eval:
            logger.error("No matching systems found. Available: %s", list(all_systems.keys()))
            return
    else:
        systems_to_eval = all_systems

    logger.info("Systems to evaluate: %s", list(systems_to_eval.keys()))

    # ── Run evaluations ─────────────────────────────────────────────────────
    reports: List[EvaluationReport] = []
    start_time = time.time()

    for i, (system_name, build_fn) in enumerate(systems_to_eval.items()):
        logger.info("")
        logger.info("=" * 60)
        logger.info("[%d/%d] Evaluating: %s", i + 1, len(systems_to_eval), system_name)
        logger.info("=" * 60)

        system_start = time.time()
        report = evaluate_system(system_name, build_fn, samples, evaluator, resume=resume)
        system_elapsed = time.time() - system_start

        reports.append(report)
        print(report.summary())
        logger.info("[%s] Completed in %.1f seconds.", system_name, system_elapsed)

    total_elapsed = time.time() - start_time

    # ── Final comparison ────────────────────────────────────────────────────
    print("\n")
    print("=" * 80)
    print("  COMPREHENSIVE EVALUATION RESULTS")
    print(f"  {len(samples)} samples × {len(reports)} systems")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print("=" * 80)
    print(evaluator.compare_reports(reports))

    # ── Save comparison table as CSV ────────────────────────────────────────
    comparison_csv = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "system", "num_samples", "exact_match", "f1", "rouge_l",
            "bleu_1", "mrr", "retrieval_precision",
        ])
        writer.writeheader()
        for r in reports:
            writer.writerow({
                "system": r.system_name,
                "num_samples": r.num_samples,
                "exact_match": round(r.avg_exact_match, 4),
                "f1": round(r.avg_f1, 4),
                "rouge_l": round(r.avg_rouge_l, 4),
                "bleu_1": round(r.avg_bleu_1, 4),
                "mrr": round(r.avg_mrr, 4),
                "retrieval_precision": round(r.avg_retrieval_precision, 4),
            })
    print(f"\nComparison summary saved: {comparison_csv}")

    # ── Save charts ─────────────────────────────────────────────────────────
    try:
        viz = MatplotlibVisualizer()
        chart_path = os.path.join(RESULTS_DIR, "full_comparison.png")
        viz.pipeline_comparison(reports, title="WEAR-RAG Comprehensive Comparison", save_path=chart_path)
        print(f"Comparison chart saved:  {chart_path}")

        metrics_path = os.path.join(RESULTS_DIR, "full_metrics.png")
        viz.metric_breakdown(reports, title="All Metrics — Full Dataset", save_path=metrics_path)
        print(f"Metrics chart saved:     {metrics_path}")
    except Exception as e:
        logger.warning("Could not save charts: %s", e)

    print(f"\nAll results in: {RESULTS_DIR}/")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WEAR-RAG Comprehensive Full-Dataset Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_eval.py --samples 10 --mock          # Quick test
  python run_full_eval.py --full --mock                 # Full dataset, mock LLM
  python run_full_eval.py --full                        # Full dataset, real LLM
  python run_full_eval.py --full --resume               # Resume interrupted run
  python run_full_eval.py --samples 500 --systems "Naive RAG (BM25),WEAR-RAG (Full)"
"""
    )
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of HotpotQA samples (default: 100)")
    parser.add_argument("--full", action="store_true",
                        help="Use ALL HotpotQA validation samples (~7405)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM (no Ollama needed)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint files")
    parser.add_argument("--systems", type=str, default=None,
                        help="Comma-separated list of systems to evaluate")
    args = parser.parse_args()

    n_samples = 0 if args.full else args.samples
    selected = [s.strip() for s in args.systems.split(",")] if args.systems else None

    run_full_evaluation(
        n_samples=n_samples,
        use_mock=args.mock,
        resume=args.resume,
        selected_systems=selected,
    )


if __name__ == "__main__":
    main()
