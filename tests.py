"""
WEAR-RAG — Unit Tests
======================
Tests all pipeline components without requiring Ollama or GPU.

Run with:
    pytest tests.py -v
    pytest tests.py -v --cov=. --cov-report=term-missing
"""

import math
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


# ===========================================================================
# Document Processor
# ===========================================================================

class TestSemanticChunker:

    def _make_chunker(self):
        """Instantiate chunker with a tiny mock model to avoid downloading."""
        from unittest.mock import MagicMock, patch
        from config import ChunkingConfig
        from document_processor import SemanticChunker

        # Mock the sentence encoder so tests run without downloading BAAI/bge-small-en
        mock_model = MagicMock()
        # Return deterministic random embeddings
        def mock_encode(sentences, **kwargs):
            rng = np.random.RandomState(len(sentences))
            emb = rng.randn(len(sentences), 384).astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            return emb / norms
        mock_model.encode.side_effect = mock_encode

        config = ChunkingConfig(similarity_threshold=0.5, min_chunk_size=3, max_chunk_size=50)
        return SemanticChunker(config, embedding_model=mock_model)

    def test_sentence_splitter_basic(self):
        from document_processor import SemanticChunker
        sentences = SemanticChunker._split_sentences(
            "Hello world. This is a test. Another sentence here."
        )
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."

    def test_sentence_splitter_empty(self):
        from document_processor import SemanticChunker
        assert SemanticChunker._split_sentences("") == []

    def test_chunk_document_returns_chunks(self):
        chunker = self._make_chunker()
        text = (
            "Transformers use self-attention. "
            "Self-attention enables parallelism. "
            "RNNs process sequences step by step. "
            "This makes them slow on long inputs."
        )
        chunks = chunker.chunk_document(text, "doc_1")
        assert len(chunks) >= 1
        # All text should be covered
        full = " ".join(c.text for c in chunks)
        # Every sentence from the original should appear
        assert "Transformers" in full

    def test_chunk_ids_are_unique(self):
        chunker = self._make_chunker()
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunker.chunk_document(text, "doc_x")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_corpus_chunking(self):
        chunker = self._make_chunker()
        docs = [
            {"id": "d1", "text": "Apple is a fruit. Oranges are citrus. Bananas are yellow."},
            {"id": "d2", "text": "Cars have four wheels. Trucks are larger. Motorcycles have two wheels."},
        ]
        all_chunks = chunker.chunk_corpus(docs)
        assert len(all_chunks) >= 2
        doc_ids = {c.source_doc_id for c in all_chunks}
        assert "d1" in doc_ids and "d2" in doc_ids


# ===========================================================================
# Evaluator metrics
# ===========================================================================

class TestMetrics:

    def test_exact_match_true(self):
        from evaluator import exact_match
        assert exact_match("London", "london") == 1.0
        assert exact_match("the cat", "cat") == 1.0   # article removed

    def test_exact_match_false(self):
        from evaluator import exact_match
        assert exact_match("Paris", "London") == 0.0

    def test_f1_perfect(self):
        from evaluator import token_f1
        assert token_f1("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)

    def test_f1_partial(self):
        from evaluator import token_f1
        score = token_f1("quick fox", "the quick brown fox")
        assert 0.0 < score < 1.0

    def test_f1_empty(self):
        from evaluator import token_f1
        assert token_f1("", "") == 1.0
        assert token_f1("answer", "") == 0.0

    def test_retrieval_precision_all_relevant(self):
        from evaluator import retrieval_precision
        assert retrieval_precision(["a", "b"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_retrieval_precision_none_relevant(self):
        from evaluator import retrieval_precision
        assert retrieval_precision(["x", "y"], ["a", "b"]) == pytest.approx(0.0)

    def test_retrieval_precision_partial(self):
        from evaluator import retrieval_precision
        assert retrieval_precision(["a", "x"], ["a", "b"]) == pytest.approx(0.5)

    def test_retrieval_precision_empty(self):
        from evaluator import retrieval_precision
        assert retrieval_precision([], ["a", "b"]) == pytest.approx(0.0)


# ===========================================================================
# Weighted Evidence Aggregator
# ===========================================================================

class TestEvidenceAggregator:

    def _make_ranked_chunks(self):
        from reranker import RankedChunk
        chunks = []
        data = [
            ("c1", "doc_a", "Self-attention allows transformers to process all tokens simultaneously.", 0.90, 0.88),
            ("c2", "doc_b", "RNNs process sequences one token at a time sequentially.", 0.85, 0.80),
            ("c3", "doc_c", "GPU acceleration is possible because matrix operations are parallelisable.", 0.72, 0.65),
            ("c4", "doc_d", "Very short chunk.", 0.20, 0.15),    # should be filtered out
        ]
        for i, (cid, src, text, sim, rerank) in enumerate(data):
            rc = RankedChunk(
                chunk_id=cid,
                source_doc_id=src,
                text=text,
                similarity_score=sim,
                reranker_score=rerank,
                rank=i + 1,
            )
            chunks.append(rc)
        return chunks

    def test_weights_must_sum_to_one(self):
        from config import AggregationConfig
        from evidence_aggregator import WeightedEvidenceAggregator
        bad_config = AggregationConfig(similarity_weight=0.5, reranker_weight=0.5, density_weight=0.5)
        with pytest.raises(ValueError):
            WeightedEvidenceAggregator(bad_config)

    def test_aggregate_returns_items(self):
        from config import AggregationConfig
        from evidence_aggregator import WeightedEvidenceAggregator
        config = AggregationConfig(score_threshold=0.3)
        aggregator = WeightedEvidenceAggregator(config)
        chunks = self._make_ranked_chunks()
        items = aggregator.aggregate(chunks)
        assert len(items) >= 1

    def test_items_sorted_by_score(self):
        from config import AggregationConfig
        from evidence_aggregator import WeightedEvidenceAggregator
        config = AggregationConfig(score_threshold=0.0)
        aggregator = WeightedEvidenceAggregator(config)
        items = aggregator.aggregate(self._make_ranked_chunks())
        scores = [item.evidence_score for item in items]
        assert scores == sorted(scores, reverse=True)

    def test_low_score_filtered(self):
        from config import AggregationConfig
        from evidence_aggregator import WeightedEvidenceAggregator
        # High threshold should filter all chunks
        config = AggregationConfig(score_threshold=0.99)
        aggregator = WeightedEvidenceAggregator(config)
        items = aggregator.aggregate(self._make_ranked_chunks())
        assert items == []

    def test_evidence_ranks_assigned(self):
        from config import AggregationConfig
        from evidence_aggregator import WeightedEvidenceAggregator
        config = AggregationConfig(score_threshold=0.0)
        aggregator = WeightedEvidenceAggregator(config)
        items = aggregator.aggregate(self._make_ranked_chunks())
        for i, item in enumerate(items):
            assert item.evidence_rank == i + 1

    def test_information_density(self):
        from evidence_aggregator import WeightedEvidenceAggregator
        # Longer, richer text should score higher than a trivially short one
        long_text = (
            "The Transformer architecture introduced in 'Attention is All You Need' "
            "by Vaswani et al. fundamentally changed Natural Language Processing. "
            "Its self-attention mechanism enables the model to weigh contextual relationships "
            "across the entire input sequence simultaneously."
        )
        short_text = "OK."
        long_score  = WeightedEvidenceAggregator._information_density(long_text)
        short_score = WeightedEvidenceAggregator._information_density(short_text)
        assert long_score > short_score

    def test_token_budget_trim(self):
        from config import AggregationConfig
        from evidence_aggregator import WeightedEvidenceAggregator
        config = AggregationConfig(score_threshold=0.0)
        aggregator = WeightedEvidenceAggregator(config)
        # Very tight budget should reduce items
        items_full    = aggregator.aggregate(self._make_ranked_chunks())
        items_trimmed = aggregator.aggregate(self._make_ranked_chunks(), token_budget=5)
        assert len(items_trimmed) <= len(items_full)

    def test_build_context_string(self):
        from config import AggregationConfig
        from evidence_aggregator import WeightedEvidenceAggregator
        config = AggregationConfig(score_threshold=0.0)
        aggregator = WeightedEvidenceAggregator(config)
        items = aggregator.aggregate(self._make_ranked_chunks())
        context = aggregator.build_context(items)
        assert isinstance(context, str)
        assert len(context) > 0
        assert "[Evidence 1" in context


# ===========================================================================
# Query Decomposer
# ===========================================================================

class TestQueryDecomposer:

    def test_always_includes_original(self):
        from query_decomposer import RuleBasedDecomposer
        d = RuleBasedDecomposer()
        result = d.decompose("What is machine learning?")
        assert "What is machine learning?" in result

    def test_comparison_decomposition(self):
        from query_decomposer import RuleBasedDecomposer
        d = RuleBasedDecomposer()
        result = d.decompose("Why are transformers better than RNNs?")
        assert len(result) > 1

    def test_no_duplicate_queries(self):
        from query_decomposer import RuleBasedDecomposer
        d = RuleBasedDecomposer()
        result = d.decompose("What is deep learning and neural networks?")
        lower = [q.lower().strip() for q in result]
        assert len(lower) == len(set(lower))

    def test_max_sub_queries_respected(self):
        from query_decomposer import RuleBasedDecomposer
        d = RuleBasedDecomposer(max_sub_queries=2)
        result = d.decompose("Why are transformers better than RNNs for NLP tasks?")
        assert len(result) <= 2


# ===========================================================================
# Run all tests
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
