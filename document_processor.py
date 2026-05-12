"""
WEAR-RAG — Document Processor
==============================
Implements semantic chunking: splits documents based on meaning shifts
between sentences rather than by fixed character or token count.

Why semantic chunking?
  Fixed-length splits arbitrarily cut mid-context.
  Semantic splits preserve topic coherence within each chunk,
  which improves retrieval relevance and answer quality.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import ChunkingConfig

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single semantically coherent chunk of a document."""
    chunk_id: str
    source_doc_id: str
    text: str
    sentences: List[str]
    start_sentence_idx: int
    end_sentence_idx: int
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)


class SemanticChunker:
    """
    Splits documents into semantically coherent chunks.

    Algorithm:
        1. Sentence-tokenize the document.
        2. Embed each sentence using a lightweight encoder.
        3. Compute cosine similarity between adjacent sentence embeddings.
        4. Insert a chunk boundary where similarity drops below the threshold,
           indicating a topic shift.
        5. Enforce min/max chunk sizes to avoid degenerate chunks.
        6. Optionally overlap boundary sentences to preserve cross-chunk context.
    """

    def __init__(self, config: ChunkingConfig, embedding_model: Optional[SentenceTransformer] = None):
        self.config = config
        # Re-use an existing model if provided (avoids double-loading)
        if embedding_model is not None:
            self.model = embedding_model
        else:
            logger.info("Loading sentence encoder for semantic chunking...")
            self.model = SentenceTransformer("BAAI/bge-small-en")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, text: str, doc_id: str) -> List[DocumentChunk]:
        """
        Split *text* into semantic chunks.

        Args:
            text:   Raw document text.
            doc_id: Identifier for the source document.

        Returns:
            List of DocumentChunk objects.
        """
        sentences = self._split_sentences(text)
        if len(sentences) == 0:
            return []

        if len(sentences) == 1:
            return [self._make_chunk(doc_id, 0, sentences, 0, 0)]

        # Embed all sentences in one pass
        embeddings = self.model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

        # Compute cosine similarity between consecutive sentences
        similarities = self._adjacent_similarities(embeddings)

        # Identify split points
        split_indices = self._find_split_points(sentences, similarities)

        # Build chunks from split points
        chunks = self._build_chunks(doc_id, sentences, split_indices)
        logger.debug("doc_id=%s  sentences=%d  chunks=%d", doc_id, len(sentences), len(chunks))
        return chunks

    def chunk_corpus(self, documents: List[dict]) -> List[DocumentChunk]:
        """
        Chunk a list of documents.

        Args:
            documents: List of dicts with keys 'id' and 'text'.

        Returns:
            Flat list of all DocumentChunk objects.
        """
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            chunks = self.chunk_document(doc["text"], doc["id"])
            all_chunks.extend(chunks)
        logger.info("Corpus chunked: %d documents → %d chunks", len(documents), len(all_chunks))
        return all_chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Lightweight rule-based sentence splitter."""
        # Split on sentence-ending punctuation followed by whitespace + capital
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        raw = re.split(pattern, text.strip())
        # Remove empty strings and strip whitespace
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _adjacent_similarities(embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between each consecutive pair of sentence embeddings.
        Assumes embeddings are already L2-normalized (dot product = cosine similarity).
        """
        # embeddings shape: (n_sentences, dim)
        return np.sum(embeddings[:-1] * embeddings[1:], axis=1)

    def _find_split_points(self, sentences: List[str], similarities: np.ndarray) -> List[int]:
        """
        Return indices of sentence boundaries where a new chunk should start.

        A boundary is placed where:
            (a) similarity falls below the threshold  OR
            (b) the running chunk would exceed max_chunk_size tokens.
        """
        split_points: List[int] = [0]      # every chunk starts after a split point
        current_size = len(sentences[0].split())

        for i, sim in enumerate(similarities):
            next_sentence_size = len(sentences[i + 1].split())
            topic_shift = sim < self.config.similarity_threshold
            would_overflow = (current_size + next_sentence_size) > self.config.max_chunk_size

            if topic_shift or would_overflow:
                # Only split if current chunk meets minimum size
                if current_size >= self.config.min_chunk_size:
                    split_points.append(i + 1)
                    current_size = next_sentence_size
                    continue

            current_size += next_sentence_size

        return split_points

    def _build_chunks(
        self, doc_id: str, sentences: List[str], split_points: List[int]
    ) -> List[DocumentChunk]:
        """Assemble DocumentChunk objects from split points."""
        chunks: List[DocumentChunk] = []
        n = len(sentences)

        for idx, start in enumerate(split_points):
            end = split_points[idx + 1] if idx + 1 < len(split_points) else n

            # Optional sentence overlap: include the last sentence of the previous chunk
            effective_start = max(0, start - self.config.overlap_sentences) if idx > 0 else start

            chunk_sentences = sentences[effective_start:end]
            chunks.append(self._make_chunk(doc_id, idx, chunk_sentences, effective_start, end - 1))

        return chunks

    @staticmethod
    def _make_chunk(
        doc_id: str,
        chunk_idx: int,
        sentences: List[str],
        start_idx: int,
        end_idx: int,
    ) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=f"{doc_id}_chunk_{chunk_idx}",
            source_doc_id=doc_id,
            text=" ".join(sentences),
            sentences=sentences,
            start_sentence_idx=start_idx,
            end_sentence_idx=end_idx,
        )
