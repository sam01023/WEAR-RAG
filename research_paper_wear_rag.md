# WEAR-RAG: Weighted Evidence Aggregation for Multi-Hop Retrieval-Augmented Generation

## Abstract
Retrieval-Augmented Generation (RAG) improves factual grounding by retrieving context before generation, but standard pipelines often over-trust top-k retrieval and pass noisy chunks to the language model. This project presents WEAR-RAG, a modular RAG pipeline that combines semantic chunking, query decomposition, cross-encoder reranking, and weighted evidence aggregation. The key idea is to compute a composite evidence score for each candidate chunk by combining dense-retrieval similarity, cross-encoder relevance, and an information-density heuristic, then filtering and ranking evidence before generation. We evaluate on 100 HotpotQA distractor-validation samples and compare against a baseline RAG implementation. WEAR-RAG improves Exact Match (0.08 vs 0.07), F1 (0.2339 vs 0.2136), ROUGE-L (0.2330 vs 0.2123), BLEU-1 (0.1733 vs 0.1582), and retrieval precision (0.3960 vs 0.3480), with near-identical MRR. The project also includes a Flask web application and mock mode for lightweight demonstration and reproducible testing.

Keywords: Retrieval-Augmented Generation, Multi-hop QA, Evidence Aggregation, Reranking, HotpotQA

## 1. Introduction
Large language models are strong at fluent generation but can hallucinate when evidence is weak or noisy. RAG reduces hallucination by injecting retrieved passages into the prompt, but three practical issues remain:

1. Fixed chunking can split semantically coherent content.
2. Single-query retrieval can miss multi-hop evidence.
3. Equal treatment of retrieved chunks can introduce noise into generation.

This project addresses these issues with WEAR-RAG (Weighted Evidence Aggregation for RAG), designed as an end-to-end, production-ready pipeline with reusable model loading, evaluation tooling, and a deployable web interface.

## 2. Problem Statement
Given a question $q$ and a corpus $D$, produce an answer $a$ that is both relevant and grounded in retrieved evidence. The challenge is to maximize answer quality while minimizing irrelevant context passed to the generator.

In standard RAG, retrieval quality and generation quality are tightly coupled. If retrieval returns mixed-quality chunks, generation degrades even with a capable LLM.

## 3. Proposed Method: WEAR-RAG

### 3.1 Pipeline Overview
WEAR-RAG uses the following stages:

1. Semantic chunking of documents.
2. Query decomposition into sub-queries.
3. Dense retrieval using FAISS and BGE embeddings.
4. Cross-encoder reranking of candidates.
5. Weighted evidence aggregation and threshold filtering.
6. Grounded answer generation via Ollama-hosted Mistral.

### 3.2 Semantic Chunking
Documents are split by sentence-level semantic shifts instead of fixed windows. Adjacent sentence embeddings are compared, and chunk boundaries are inserted when similarity drops below a threshold or max length is exceeded.

Configured defaults:

- similarity threshold: 0.75
- min chunk size: 50 tokens
- max chunk size: 300 tokens
- overlap: 1 sentence

### 3.3 Query Decomposition
A rule-based decomposer expands complex questions into focused sub-queries while always retaining the original query. For example, comparative questions can produce prompts for entity definition, advantages, and limitations. An optional LLM decomposer exists with rule-based fallback.

### 3.4 Retrieval and Reranking
Dense retrieval uses BAAI/bge-small-en embeddings and FAISS IndexFlatIP on normalized vectors. Candidates are reranked with BAAI/bge-reranker-base (cross-encoder), and logits are mapped with sigmoid to $[0,1]$.

### 3.5 Weighted Evidence Aggregation (Core Contribution)
Each chunk receives a composite score:

$$
	ext{EvidenceScore} = w_{sim}\cdot s_{sim} + w_{rank}\cdot s_{rank} + w_{dens}\cdot s_{dens}
$$

Where:

- $s_{sim}$ is dense retrieval similarity
- $s_{rank}$ is cross-encoder reranker score
- $s_{dens}$ is information density heuristic

Default weights:

- $w_{sim}=0.5$
- $w_{rank}=0.4$
- $w_{dens}=0.1$

Only chunks with score >= 0.3 are kept, then trimmed by token budget before generation.

### 3.6 Generation
The generator uses an evidence-constrained prompt. If evidence is insufficient, the model is instructed to answer "I don't know" instead of fabricating. A mock generator is provided for offline tests.

## 4. System Implementation

### 4.1 Core Modules
- `main.py`: orchestrates Baseline RAG, Improved RAG, and WEAR-RAG pipelines.
- `document_processor.py`: semantic chunking.
- `embeddings.py`: dense embedding engine.
- `vector_store.py`: FAISS-backed retrieval.
- `query_decomposer.py`: query decomposition.
- `reranker.py`: cross-encoder reranking.
- `evidence_aggregator.py`: weighted evidence scoring and filtering.
- `llm_generator.py`: Ollama and mock generation backends.
- `evaluator.py`: dataset loading and metric computation.
- `visualizer.py`: ASCII and matplotlib reporting.

### 4.2 Evaluation and Outputs
Evaluation mode saves:

- per-sample CSV reports (`results_baseline_rag.csv`, `results_wear-rag.csv`)
- comparison plots (`wear_rag_comparison.png`, `wear_rag_metrics.png`)

### 4.3 Web Application and Demo Support
The project includes Flask backends for interactive QA and upload handling:

- `app.py` (root web app)
- `wear_rag_webapp/app.py` (alternate web app path)

Endpoints include health checks, file upload (`.txt` and `.pdf`), and QA inference. A lightweight `mock_pipeline.py` supports no-heavy-model demos.

## 5. Experimental Setup

### 5.1 Dataset
- Dataset: HotpotQA
- Configuration: distractor
- Split: validation
- Samples: 100

### 5.2 Compared Systems
1. Baseline RAG: fixed chunking, single retrieval, no reranker, no weighted aggregation.
2. WEAR-RAG: full proposed pipeline.

### 5.3 Metrics
The evaluator reports:

- Exact Match (EM)
- Token-level F1
- ROUGE-L
- BLEU-1
- Mean Reciprocal Rank (MRR)
- Retrieval Precision

### 5.4 Runtime Configuration
- Embedding model: BAAI/bge-small-en
- Reranker: BAAI/bge-reranker-base
- LLM endpoint: Ollama (`mistral`)
- Context window: 4096

## 6. Results

### 6.1 Main Quantitative Results (n=100)

| Metric | Baseline RAG | WEAR-RAG | Absolute Delta | Relative Delta |
|---|---:|---:|---:|---:|
| Exact Match | 0.0700 | 0.0800 | +0.0100 | +14.29% |
| F1 | 0.2136 | 0.2339 | +0.0203 | +9.50% |
| ROUGE-L | 0.2123 | 0.2330 | +0.0207 | +9.75% |
| BLEU-1 | 0.1582 | 0.1733 | +0.0151 | +9.54% |
| MRR | 0.9558 | 0.9550 | -0.0008 | -0.08% |
| Retrieval Precision | 0.3480 | 0.3960 | +0.0480 | +13.79% |

### 6.2 Interpretation
WEAR-RAG improves answer-level quality metrics and retrieval precision, indicating better evidence selection rather than just ranking-first-hit effects. Nearly unchanged MRR suggests both systems often retrieve at least one relevant document early, while WEAR-RAG better selects final context for generation.

## 7. Ablation Perspective
Although only baseline vs WEAR-RAG is benchmarked in the included CSVs, code structure supports ablations for:

1. no decomposition
2. no reranker
3. no density term
4. no score thresholding
5. fixed chunking vs semantic chunking

These can be added by extending the `systems` list in evaluation mode and re-running sample batches.

## 8. Reproducibility

### 8.1 Environment
The root `requirements.txt` currently includes only Flask, while the full pipeline uses additional libraries (FAISS, sentence-transformers, datasets, numpy, matplotlib, requests, and optional PDF parsers). For strict reproducibility, a complete dependency lock file is recommended.

### 8.2 Typical Evaluation Command
From project root:

```bash
python main.py --mode evaluate --samples 100
```

Mock mode (no Ollama):

```bash
python main.py --mode evaluate --samples 100 --mock
```

### 8.3 Test Coverage
`tests.py` includes unit tests for chunking behavior, metrics, aggregation scoring constraints, token budget trimming, and decomposition behavior.

## 9. Practical Contributions
This project contributes:

1. A complete WEAR-RAG implementation with modular components.
2. A weighted evidence scoring framework that improves QA quality.
3. Model registry optimization that avoids repeated heavy model loads.
4. Built-in evaluator and visualization toolkit.
5. Deployable Flask web application and mock pipeline for demonstration.

## 10. Limitations
1. Single-dataset evaluation (HotpotQA only) limits generalization claims.
2. No statistical significance testing is currently reported.
3. Root dependency specification is incomplete for one-command setup.
4. LLM generation quality may vary by local Ollama model and hardware.

## 11. Future Work
1. Add broader datasets (e.g., NQ, TriviaQA, domain-specific corpora).
2. Perform full ablation and statistical significance tests.
3. Learn aggregation weights instead of fixed manual weights.
4. Add multilingual support and domain adaptation.
5. Integrate long-context reranking and citation-level grounding.

## 12. Conclusion
WEAR-RAG demonstrates that retrieval quality alone is insufficient; evidence weighting and filtering before generation materially improve answer quality. On 100 HotpotQA validation samples, the proposed approach yields consistent gains over baseline RAG across EM, F1, ROUGE-L, BLEU-1, and retrieval precision. The implementation is practical, modular, and deployment-ready, making it suitable for both academic reporting and applied QA systems.

## Appendix A: Project Artifact Map

- Core pipeline: `main.py`
- Configuration: `config.py`
- Chunking: `document_processor.py`
- Embeddings: `embeddings.py`
- Vector retrieval: `vector_store.py`
- Query decomposition: `query_decomposer.py`
- Reranking: `reranker.py`
- Evidence aggregation: `evidence_aggregator.py`
- Generation: `llm_generator.py`
- Evaluation: `evaluator.py`
- Visualizations: `visualizer.py`
- Tests: `tests.py`
- Results CSVs: `results_baseline_rag.csv`, `results_wear-rag.csv`
- Charts: `wear_rag_comparison.png`, `wear_rag_metrics.png`
- Web app: `app.py`, `wear_rag_webapp/app.py`

## Appendix B: Ready-to-Submit Paper Skeleton
For conference formatting (IEEE/ACM/Springer), map sections as:

1. Abstract
2. Introduction
3. Related Work
4. Method
5. Experimental Setup
6. Results and Discussion
7. Limitations and Ethics
8. Conclusion
9. References

The content in this document can be directly transferred into LaTeX with minimal edits.
