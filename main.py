"""
WEAR-RAG — Main Pipeline (v2 — Production Ready)
=================================================
Key improvements over v1:
    - Models loaded ONCE and shared across all pipelines (3x faster evaluation)
    - Noisy httpx / HuggingFace logs suppressed
    - ImprovedRAG now shows measurable improvement over Baseline via
      diversity-aware context selection across sub-queries
    - Results saved to CSV automatically
    - Cleaner CLI with --samples flag

Usage:
    python main.py --mode demo
    python main.py --mode demo --mock
    python main.py --mode evaluate --samples 10
    python main.py --mode evaluate --samples 10 --mock
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

# ── Suppress noisy third-party loggers before anything else ────────────────
for _noisy in ("httpx", "httpcore", "huggingface_hub", "huggingface_hub.utils._http",
               "sentence_transformers", "transformers", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wear_rag.main")

# ── Project imports ────────────────────────────────────────────────────────
from config import WEARRAGConfig, DEFAULT_CONFIG
from document_processor import SemanticChunker, DocumentChunk
from embeddings import EmbeddingEngine
from vector_store import VectorStore, RetrievedChunk
from query_decomposer import build_decomposer
from reranker import Reranker, RankedChunk
from evidence_aggregator import WeightedEvidenceAggregator, EvidenceItem
from llm_generator import build_generator
from evaluator import Evaluator, EvaluationReport
from visualizer import ASCIIVisualizer, MatplotlibVisualizer


# ===========================================================================
# Shared Model Registry  (load once, reuse everywhere)
# ===========================================================================

class ModelRegistry:
    """
    Loads heavy models once and provides them to all pipeline instances.
    Prevents reloading BAAI/bge-small-en and BAAI/bge-reranker-base
    for every pipeline variant during evaluation.
    """

    def __init__(self, config: WEARRAGConfig):
        self.config = config
        self._embedding_engine: Optional[EmbeddingEngine] = None
        self._reranker: Optional[Reranker] = None
        self._chunker: Optional[SemanticChunker] = None

    @property
    def embedding_engine(self) -> EmbeddingEngine:
        if self._embedding_engine is None:
            logger.info("Loading embedding model (once)...")
            self._embedding_engine = EmbeddingEngine(self.config.embedding)
        return self._embedding_engine

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            logger.info("Loading reranker model (once)...")
            self._reranker = Reranker(self.config.retrieval)
        return self._reranker

    @property
    def chunker(self) -> SemanticChunker:
        if self._chunker is None:
            self._chunker = SemanticChunker(
                self.config.chunking,
                self.embedding_engine.underlying_model
            )
        return self._chunker


# ===========================================================================
# Pipeline Implementations
# ===========================================================================

class BaselineRAG:
    """
    Standard RAG — fixed chunking, single retrieval, no reranking.
    All retrieved chunks treated equally.
    """

    def __init__(self, registry: ModelRegistry, use_mock_llm: bool = False, top_k: int = 5):
        self.config = registry.config
        self.embedding_engine = registry.embedding_engine
        self.vector_store = VectorStore(self.embedding_engine, "./vs_baseline")
        self.generator = build_generator(use_mock=use_mock_llm, model=self.config.llm.model_name)
        self.top_k = top_k

    def ingest(self, documents: List[dict]) -> None:
        chunks = self._fixed_chunk(documents)
        self.vector_store.add_chunks(chunks)

    def answer(self, question: str) -> Tuple[str, List[str], List[float]]:
        results = self.vector_store.search(question, top_k=self.top_k)
        evidence_items = [
            EvidenceItem(
                chunk_id=r.chunk.chunk_id,
                source_doc_id=r.chunk.source_doc_id,
                text=r.chunk.text,
                similarity_score=r.similarity_score,
                reranker_score=0.0,
                density_score=0.0,
                evidence_score=r.similarity_score,
                evidence_rank=i + 1,
            )
            for i, r in enumerate(results)
        ]
        answer = self.generator.generate(question, evidence_items)
        return answer, [e.source_doc_id for e in evidence_items], [e.evidence_score for e in evidence_items]

    @staticmethod
    def _fixed_chunk(documents: List[dict], chunk_size: int = 300) -> List[DocumentChunk]:
        chunks = []
        for doc in documents:
            words = doc["text"].split()
            for i in range(0, max(1, len(words)), chunk_size):
                piece = " ".join(words[i:i + chunk_size])
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc['id']}_chunk_{i // chunk_size}",
                    source_doc_id=doc["id"],
                    text=piece,
                    sentences=[piece],
                    start_sentence_idx=i,
                    end_sentence_idx=i + chunk_size,
                ))
        return chunks


class NaiveRAG:
    """
    Naive RAG — BM25 keyword retrieval, fixed chunking, no reranking.
    The simplest possible RAG: pure keyword matching.
    """

    def __init__(self, registry: ModelRegistry, use_mock_llm: bool = False, top_k: int = 5):
        self.config = registry.config
        self.generator = build_generator(use_mock=use_mock_llm, model=self.config.llm.model_name)
        self.top_k = top_k
        self._chunks: List[DocumentChunk] = []
        self._bm25 = None

    def ingest(self, documents: List[dict]) -> None:
        self._chunks = BaselineRAG._fixed_chunk(documents)
        # Build BM25 index
        tokenised = [chunk.text.lower().split() for chunk in self._chunks]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(tokenised)
        except ImportError:
            # Fallback: simple TF-IDF-like scoring
            self._bm25 = None

    def answer(self, question: str) -> Tuple[str, List[str], List[float]]:
        q_tokens = question.lower().split()
        if self._bm25 is not None:
            scores = self._bm25.get_scores(q_tokens)
        else:
            # Fallback: word overlap ratio
            q_set = set(q_tokens)
            scores = []
            for chunk in self._chunks:
                c_tokens = set(chunk.text.lower().split())
                overlap = len(q_set & c_tokens) / max(len(q_set), 1)
                scores.append(overlap)

        # Get top-k
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:self.top_k]

        evidence_items = []
        for rank, idx in enumerate(top_indices):
            chunk = self._chunks[idx]
            evidence_items.append(EvidenceItem(
                chunk_id=chunk.chunk_id,
                source_doc_id=chunk.source_doc_id,
                text=chunk.text,
                similarity_score=float(scores[idx]),
                reranker_score=0.0,
                density_score=0.0,
                evidence_score=float(scores[idx]),
                evidence_rank=rank + 1,
            ))

        answer = self.generator.generate(question, evidence_items)
        return answer, [e.source_doc_id for e in evidence_items], [e.evidence_score for e in evidence_items]


class ImprovedRAG:
    """
    Improved RAG — semantic chunking + query decomposition +
    diversity-aware context selection.

    Improvement over Baseline:
        Query decomposition retrieves evidence for each sub-question independently.
        The top chunk from EACH sub-query is guaranteed a slot in the context,
        ensuring multi-hop coverage rather than all chunks being about one aspect.
    """

    def __init__(self, registry: ModelRegistry, use_mock_llm: bool = False):
        self.config = registry.config
        self.embedding_engine = registry.embedding_engine
        self.chunker = registry.chunker
        self.vector_store = VectorStore(self.embedding_engine, "./vs_improved")
        self.decomposer = build_decomposer(use_llm=False)
        self.generator = build_generator(use_mock=use_mock_llm, model=self.config.llm.model_name)

    def ingest(self, documents: List[dict]) -> None:
        chunks = self.chunker.chunk_corpus(documents)
        self.vector_store.add_chunks(chunks)

    def answer(self, question: str) -> Tuple[str, List[str], List[float]]:
        sub_queries = self.decomposer.decompose(question)

        # Retrieve per sub-query
        results_per_query: List[List[RetrievedChunk]] = []
        for sq in sub_queries:
            results_per_query.append(
                self.vector_store.search(sq, top_k=self.config.retrieval.top_k_retrieval)
            )

        # Diversity-aware selection: guarantee top chunk from each sub-query,
        # then fill remaining slots with next-best unique chunks
        seen_ids = set()
        selected: List[RetrievedChunk] = []
        max_per_query = max(1, self.config.retrieval.top_k_rerank // len(sub_queries))

        # First pass: top chunk from each sub-query
        for results in results_per_query:
            for r in results:
                if r.chunk.chunk_id not in seen_ids:
                    selected.append(r)
                    seen_ids.add(r.chunk.chunk_id)
                    break

        # Second pass: fill remaining slots with best unique chunks
        all_remaining = [
            r for results in results_per_query for r in results
            if r.chunk.chunk_id not in seen_ids
        ]
        all_remaining.sort(key=lambda x: x.similarity_score, reverse=True)
        for r in all_remaining:
            if len(selected) >= self.config.retrieval.top_k_rerank:
                break
            if r.chunk.chunk_id not in seen_ids:
                selected.append(r)
                seen_ids.add(r.chunk.chunk_id)

        evidence_items = [
            EvidenceItem(
                chunk_id=r.chunk.chunk_id,
                source_doc_id=r.chunk.source_doc_id,
                text=r.chunk.text,
                similarity_score=r.similarity_score,
                reranker_score=0.0,
                density_score=0.0,
                evidence_score=r.similarity_score,
                evidence_rank=i + 1,
            )
            for i, r in enumerate(selected)
        ]
        answer = self.generator.generate(question, evidence_items)
        return answer, [e.source_doc_id for e in evidence_items], [e.evidence_score for e in evidence_items]


class RerankOnlyRAG:
    """
    Rerank-Only RAG — semantic chunking + vector retrieval + cross-encoder reranking.
    No query decomposition, no weighted evidence aggregation.
    Uses reranker score directly as evidence score.
    Isolates the reranking contribution.
    """

    def __init__(self, registry: ModelRegistry, use_mock_llm: bool = False):
        self.config = registry.config
        self.embedding_engine = registry.embedding_engine
        self.chunker = registry.chunker
        self.vector_store = VectorStore(self.embedding_engine, "./vs_rerank_only")
        self.reranker = registry.reranker
        self.generator = build_generator(use_mock=use_mock_llm, model=self.config.llm.model_name)

    def ingest(self, documents: List[dict]) -> None:
        chunks = self.chunker.chunk_corpus(documents)
        self.vector_store.add_chunks(chunks)

    def answer(self, question: str) -> Tuple[str, List[str], List[float]]:
        # Single query retrieval (no decomposition)
        candidates = self.vector_store.search(question, top_k=self.config.retrieval.top_k_retrieval)
        ranked_chunks = self.reranker.rerank(question, candidates,
                                              top_k=self.config.retrieval.top_k_rerank)

        evidence_items = [
            EvidenceItem(
                chunk_id=rc.chunk_id,
                source_doc_id=rc.source_doc_id,
                text=rc.text,
                similarity_score=rc.similarity_score,
                reranker_score=rc.reranker_score,
                density_score=0.0,
                evidence_score=rc.reranker_score,   # reranker score only
                evidence_rank=rc.rank,
            )
            for rc in ranked_chunks
        ]

        answer = self.generator.generate(question, evidence_items)
        return answer, [e.source_doc_id for e in evidence_items], [e.evidence_score for e in evidence_items]


class HybridRAG:
    """
    Hybrid Retrieval RAG — combines BM25 (sparse) + Dense (FAISS) retrieval
    using Reciprocal Rank Fusion (RRF) to merge results.
    Semantic chunking, no reranking, no query decomposition.
    """

    def __init__(self, registry: ModelRegistry, use_mock_llm: bool = False, rrf_k: int = 60):
        self.config = registry.config
        self.embedding_engine = registry.embedding_engine
        self.chunker = registry.chunker
        self.vector_store = VectorStore(self.embedding_engine, "./vs_hybrid")
        self.generator = build_generator(use_mock=use_mock_llm, model=self.config.llm.model_name)
        self.rrf_k = rrf_k  # RRF constant (standard = 60)
        self._chunks: List[DocumentChunk] = []
        self._bm25 = None

    def ingest(self, documents: List[dict]) -> None:
        chunks = self.chunker.chunk_corpus(documents)
        self.vector_store.add_chunks(chunks)
        self._chunks = chunks
        # Build BM25 index on the same chunks
        tokenised = [chunk.text.lower().split() for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(tokenised)
        except ImportError:
            self._bm25 = None

    def answer(self, question: str) -> Tuple[str, List[str], List[float]]:
        import numpy as np

        top_k = self.config.retrieval.top_k_rerank

        # Dense retrieval
        dense_results = self.vector_store.search(question, top_k=self.config.retrieval.top_k_retrieval)

        # BM25 retrieval
        q_tokens = question.lower().split()
        if self._bm25 is not None and self._chunks:
            bm25_scores = self._bm25.get_scores(q_tokens)
            bm25_top = np.argsort(bm25_scores)[::-1][:self.config.retrieval.top_k_retrieval]
            bm25_ranked = [(self._chunks[i], bm25_scores[i]) for i in bm25_top]
        else:
            bm25_ranked = []

        # Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}
        sim_map: Dict[str, float] = {}

        # Dense ranks
        for rank, r in enumerate(dense_results, 1):
            cid = r.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            chunk_map[cid] = r.chunk
            sim_map[cid] = r.similarity_score

        # BM25 ranks
        for rank, (chunk, score) in enumerate(bm25_ranked, 1):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = chunk
                sim_map[cid] = 0.0

        # Sort by RRF score and take top-k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        evidence_items = []
        for rank, cid in enumerate(sorted_ids, 1):
            chunk = chunk_map[cid]
            evidence_items.append(EvidenceItem(
                chunk_id=cid,
                source_doc_id=chunk.source_doc_id,
                text=chunk.text,
                similarity_score=sim_map.get(cid, 0.0),
                reranker_score=0.0,
                density_score=0.0,
                evidence_score=rrf_scores[cid],
                evidence_rank=rank,
            ))

        answer = self.generator.generate(question, evidence_items)
        return answer, [e.source_doc_id for e in evidence_items], [e.evidence_score for e in evidence_items]


class WEARRAG:
    """
    Full WEAR-RAG pipeline:
    Semantic chunking → Query decomposition → Cross-encoder reranking
    → Weighted evidence aggregation → LLM generation
    """

    def __init__(self, registry: ModelRegistry, use_mock_llm: bool = False,
                 aggregation_mode: str = "weighted"):
        self.config = registry.config
        self.embedding_engine = registry.embedding_engine
        self.chunker = registry.chunker
        self.vector_store = VectorStore(self.embedding_engine, "./vs_wearrag")
        self.decomposer = build_decomposer(use_llm=False)
        self.reranker = registry.reranker
        self.aggregation_mode = aggregation_mode
        self.aggregator = WeightedEvidenceAggregator(registry.config.aggregation)
        self.generator = build_generator(use_mock=use_mock_llm, model=self.config.llm.model_name)

    def ingest(self, documents: List[dict]) -> None:
        chunks = self.chunker.chunk_corpus(documents)
        self.vector_store.add_chunks(chunks)
        logger.info("Ingested %d chunks.", self.vector_store.total_chunks)

    def answer(self, question: str, verbose: bool = False) -> Tuple[str, List[str], List[float]]:
        sub_queries = self.decomposer.decompose(question)
        candidates_per_query = self.vector_store.search_multi(
            sub_queries, top_k=self.config.retrieval.top_k_retrieval
        )
        ranked_chunks = self.reranker.rerank_multi_query(sub_queries, candidates_per_query)

        if self.aggregation_mode == "weighted":
            evidence_items = self.aggregator.aggregate(
                ranked_chunks,
                token_budget=self.config.llm.context_window // 2,
            )
        elif self.aggregation_mode == "average":
            evidence_items = self._aggregate_average(ranked_chunks)
        elif self.aggregation_mode == "max":
            evidence_items = self._aggregate_max(ranked_chunks)
        else:
            evidence_items = self.aggregator.aggregate(
                ranked_chunks,
                token_budget=self.config.llm.context_window // 2,
            )

        if verbose:
            print(self.aggregator.score_summary(evidence_items))
        answer = self.generator.generate(question, evidence_items)
        return answer, [e.source_doc_id for e in evidence_items], [e.evidence_score for e in evidence_items]

    def answer_with_evidence(self, question: str) -> Tuple[str, List[EvidenceItem]]:
        sub_queries = self.decomposer.decompose(question)
        candidates_per_query = self.vector_store.search_multi(
            sub_queries, top_k=self.config.retrieval.top_k_retrieval
        )
        ranked_chunks = self.reranker.rerank_multi_query(sub_queries, candidates_per_query)
        evidence_items = self.aggregator.aggregate(ranked_chunks)
        answer = self.generator.generate(question, evidence_items)
        return answer, evidence_items

    def _aggregate_average(self, ranked_chunks) -> List[EvidenceItem]:
        """Average aggregation: score = (sim + reranker + density) / 3"""
        items = []
        for i, rc in enumerate(ranked_chunks):
            density = WeightedEvidenceAggregator._information_density(rc.text)
            score = (rc.similarity_score + rc.reranker_score + density) / 3.0
            items.append(EvidenceItem(
                chunk_id=rc.chunk_id, source_doc_id=rc.source_doc_id, text=rc.text,
                similarity_score=rc.similarity_score, reranker_score=rc.reranker_score,
                density_score=density, evidence_score=score, evidence_rank=i + 1,
            ))
        items.sort(key=lambda x: x.evidence_score, reverse=True)
        for i, item in enumerate(items):
            item.evidence_rank = i + 1
        return items[:self.config.retrieval.top_k_rerank]

    def _aggregate_max(self, ranked_chunks) -> List[EvidenceItem]:
        """Max aggregation: score = max(sim, reranker, density)"""
        items = []
        for i, rc in enumerate(ranked_chunks):
            density = WeightedEvidenceAggregator._information_density(rc.text)
            score = max(rc.similarity_score, rc.reranker_score, density)
            items.append(EvidenceItem(
                chunk_id=rc.chunk_id, source_doc_id=rc.source_doc_id, text=rc.text,
                similarity_score=rc.similarity_score, reranker_score=rc.reranker_score,
                density_score=density, evidence_score=score, evidence_rank=i + 1,
            ))
        items.sort(key=lambda x: x.evidence_score, reverse=True)
        for i, item in enumerate(items):
            item.evidence_rank = i + 1
        return items[:self.config.retrieval.top_k_rerank]


# ===========================================================================
# Demo
# ===========================================================================

def run_demo(use_mock: bool = False):
    config = DEFAULT_CONFIG
    registry = ModelRegistry(config)

    pipeline = WEARRAG(registry, use_mock_llm=use_mock)

    demo_documents = [
        {"id": "transformers_overview",
         "text": ("Transformers are deep learning models that use self-attention mechanisms. "
                  "Self-attention allows the model to weigh the importance of each token relative "
                  "to every other token in the sequence. This enables highly parallelised computation "
                  "compared to sequential RNN processing.")},
        {"id": "rnn_limitations",
         "text": ("Recurrent Neural Networks process sequences one token at a time. "
                  "This sequential nature prevents parallelisation and leads to vanishing gradient "
                  "problems on long sequences. LSTMs and GRUs partially address these issues but "
                  "still struggle with very long-range dependencies.")},
        {"id": "attention_mechanism",
         "text": ("The attention mechanism computes a weighted sum of value vectors, where weights "
                  "are derived from the compatibility between query and key vectors. Multi-head "
                  "attention applies this operation in parallel across multiple subspaces.")},
        {"id": "gpu_acceleration",
         "text": ("Transformer architectures are highly amenable to GPU acceleration because their "
                  "matrix operations can be batched across the full sequence length. RNNs require "
                  "sequential computation, making GPU utilisation far lower.")},
        {"id": "bert_gpt",
         "text": ("BERT and GPT are prominent transformer-based models. BERT is used for "
                  "understanding tasks while GPT excels at text generation. Both use self-attention "
                  "as their core mechanism.")},
    ]

    print("\n" + "=" * 62)
    print("  WEAR-RAG Demo Pipeline  (v2)")
    print("=" * 62)
    print(f"  Ingesting {len(demo_documents)} documents...")
    pipeline.ingest(demo_documents)

    question = "Why are transformers better than RNNs for sequence modelling?"
    print(f"\n  Question: {question}\n")

    answer, evidence_items = pipeline.answer_with_evidence(question)

    viz = ASCIIVisualizer()
    print(viz.evidence_chart(evidence_items))
    print(viz.score_breakdown(evidence_items))
    print(f"\n  Answer:\n  {answer}")
    print("=" * 62)


# ===========================================================================
# Evaluation
# ===========================================================================

def run_evaluation(n_samples: int = 10, use_mock: bool = False, full: bool = False):
    config = DEFAULT_CONFIG
    evaluator = Evaluator()

    actual_n = 0 if full else n_samples
    logger.info("Loading %s HotpotQA samples...", "ALL" if full else str(n_samples))
    samples = evaluator.load_hotpot_samples(n=actual_n)
    if not samples:
        logger.error("No samples loaded. Check datasets installation.")
        return

    # ── Load models ONCE ────────────────────────────────────────────────────
    logger.info("Loading shared models...")
    registry = ModelRegistry(config)
    _ = registry.embedding_engine   # trigger load
    _ = registry.reranker           # trigger load
    logger.info("Models ready.")

    reports = []
    systems = [
        ("Naive RAG (BM25)",     lambda: NaiveRAG(registry, use_mock_llm=use_mock)),
        ("Baseline RAG (k=5)",   lambda: BaselineRAG(registry, use_mock_llm=use_mock, top_k=5)),
        ("Baseline RAG (k=10)",  lambda: BaselineRAG(registry, use_mock_llm=use_mock, top_k=10)),
        ("Decomposition-Only",   lambda: ImprovedRAG(registry, use_mock_llm=use_mock)),
        ("Rerank-Only",          lambda: RerankOnlyRAG(registry, use_mock_llm=use_mock)),
        ("Hybrid Retrieval",     lambda: HybridRAG(registry, use_mock_llm=use_mock)),
        ("WEAR-RAG (Full)",      lambda: WEARRAG(registry, use_mock_llm=use_mock)),
    ]

    for system_name, build_fn in systems:
        logger.info("=" * 50)
        logger.info("Evaluating: %s", system_name)
        logger.info("=" * 50)
        pipeline = build_fn()
        csv_path = f"results_{system_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"

        def make_pipeline_fn(pl, sname):
            def fn(sample):
                # Fresh vector store per sample (each HotpotQA question has its own docs)
                if hasattr(pl, 'vector_store'):
                    pl.vector_store = VectorStore(
                        pl.embedding_engine,
                        os.path.join("/tmp", f"{sname}_{sample['id']}")
                    )
                pl.ingest(sample["documents"])
                return pl.answer(sample["question"])
            return fn

        report = evaluator.evaluate(
            make_pipeline_fn(pipeline, system_name.replace(" ", "_").lower()),
            samples,
            system_name=system_name,
            save_csv=csv_path,
        )
        reports.append(report)
        print(report.summary())

    # Final comparison
    print(evaluator.compare_reports(reports))

    # Save charts
    try:
        viz = MatplotlibVisualizer()
        viz.pipeline_comparison(reports, save_path="wear_rag_comparison.png")
        print("\nComparison chart saved: wear_rag_comparison.png")
        viz.metric_breakdown(reports, save_path="wear_rag_metrics.png")
        print("Metrics chart saved:     wear_rag_metrics.png")
    except Exception as e:
        logger.warning("Could not save charts: %s", e)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="WEAR-RAG v2")
    parser.add_argument("--mode", choices=["demo", "evaluate"], default="demo")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no Ollama needed)")
    parser.add_argument("--samples", type=int, default=10, help="HotpotQA samples for evaluation")
    parser.add_argument("--full", action="store_true", help="Use full HotpotQA validation split")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "demo":
        run_demo(use_mock=args.mock)
    elif args.mode == "evaluate":
        run_evaluation(n_samples=args.samples, use_mock=args.mock, full=args.full)
