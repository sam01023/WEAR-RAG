"""
WEAR-RAG Configuration
======================
Central configuration for all pipeline components.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en"
    batch_size: int = 32
    max_length: int = 512
    device: str = "cpu"  # change to "cuda" if GPU available


@dataclass
class ChunkingConfig:
    similarity_threshold: float = 0.75   # cosine similarity threshold for semantic split
    min_chunk_size: int = 50             # minimum tokens per chunk
    max_chunk_size: int = 300            # maximum tokens per chunk
    overlap_sentences: int = 1           # sentence overlap between chunks


@dataclass
class RetrievalConfig:
    top_k_retrieval: int = 20            # documents retrieved per sub-query
    top_k_rerank: int = 5                # documents kept after reranking
    reranker_model: str = "BAAI/bge-reranker-base"


@dataclass
class AggregationConfig:
    # Weights for evidence score: must sum to 1.0
    similarity_weight: float = 0.5
    reranker_weight: float = 0.4
    density_weight: float = 0.1
    score_threshold: float = 0.3         # minimum score to include evidence


@dataclass
class LLMConfig:
    model_name: str = "mistral"          # Ollama model name
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 512
    context_window: int = 4096


@dataclass
class EvaluationConfig:
    dataset_name: str = "hotpot_qa"
    dataset_split: str = "validation"
    num_samples: int = 100               # number of questions to evaluate; 0 = all
    full_dataset: bool = False           # if True, load ALL samples from the split
    metrics: List[str] = field(default_factory=lambda: ["exact_match", "f1", "retrieval_precision"])


@dataclass
class WEARRAGConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    vector_store_path: str = "./faiss_index"
    log_level: str = "INFO"


# Default config instance
DEFAULT_CONFIG = WEARRAGConfig()


# ---------------------------------------------------------------------------
# Cross-Model Study Configurations
# ---------------------------------------------------------------------------

def config_with_embedding(model_name: str) -> WEARRAGConfig:
    """Return a config using a different embedding model."""
    cfg = WEARRAGConfig()
    cfg.embedding = EmbeddingConfig(model_name=model_name)
    return cfg


def config_with_reranker(model_name: str) -> WEARRAGConfig:
    """Return a config using a different reranker model."""
    cfg = WEARRAGConfig()
    cfg.retrieval = RetrievalConfig(reranker_model=model_name)
    return cfg


def config_with_generator(model_name: str) -> WEARRAGConfig:
    """Return a config using a different LLM generator."""
    cfg = WEARRAGConfig()
    cfg.llm = LLMConfig(model_name=model_name)
    return cfg


# Pre-built alternative configs
ALT_EMBEDDING_CONFIG = config_with_embedding("BAAI/bge-base-en")
ALT_RERANKER_CONFIG = config_with_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
ALT_GENERATOR_CONFIG = config_with_generator("llama3")
