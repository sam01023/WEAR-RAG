"""
WEAR-RAG Web Application — Flask Backend
=========================================
Routes:
    GET  /              → serve index.html
    POST /api/ask       → run WEAR-RAG pipeline, return answer + evidence
    POST /api/upload    → parse uploaded .txt/.pdf, return extracted text
    GET  /api/health    → check Ollama + model status

Run:
    python app.py
Then open http://localhost:5000
"""

import os
import sys
import json
import uuid
import logging
import traceback
from pathlib import Path

# ── Suppress noisy loggers ────────────────────────────────────────────────
for _n in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers",
           "transformers", "filelock", "huggingface_hub.utils._http"):
    logging.getLogger(_n).setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("wear_rag.app")

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Add pipeline to path ──────────────────────────────────────────────────
# All files are in the same folder as app.py
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
logger.info("Pipeline directory: %s", PIPELINE_DIR)

from config import DEFAULT_CONFIG
from main import ModelRegistry, WEARRAG
from vector_store import VectorStore
from llm_generator import build_generator
import tempfile
import numpy as np

# ── Flask app ─────────────────────────────────────────────────────────────
_app_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(_app_dir, "static"), template_folder=os.path.join(_app_dir, "templates"))
CORS(app)

UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# ── Global model registry (loaded once on startup) ────────────────────────
logger.info("Initialising model registry...")
registry = ModelRegistry(DEFAULT_CONFIG)
# Avoid loading heavy ML models (faiss, sentence-transformers, torch) at import time.
# Allow optional eager preload via environment var `FORCE_PRELOAD=1`.
try:
    if os.environ.get("FORCE_PRELOAD", "0") == "1":
        _ = registry.embedding_engine
        _ = registry.reranker
        logger.info("Models preloaded via FORCE_PRELOAD.")
    else:
        logger.info("Skipping heavy model preload; models will load lazily on demand.")
except Exception:
    logger.warning("Model preload failed; continuing without preloaded models.", exc_info=True)


def build_pipeline(session_id: str, use_mock: bool = False, model_name: str | None = None) -> WEARRAG:
    """Create a WEAR-RAG pipeline with a session-specific vector store.

    Args:
        session_id: short id for temporary vector store path
        use_mock: if True, the pipeline will use the mock LLM generator
    """
    pipeline = WEARRAG(registry, use_mock_llm=use_mock)
    tmpdir = Path(tempfile.gettempdir())
    # For mock mode, avoid loading heavy embedding model — provide a lightweight
    # mock embedding engine compatible with VectorStore API.
    if use_mock:
        class _MockEmbeddingEngine:
            def __init__(self, dim=64):
                self.dimension = dim
            def embed_chunks(self, chunks):
                # deterministic pseudo-embeddings based on length
                arr = np.zeros((len(chunks), self.dimension), dtype=np.float32)
                for i, c in enumerate(chunks):
                    v = (len(c.text) % (self.dimension - 1)) + 1
                    arr[i, :v] = 1.0
                return arr
            def embed_query(self, query):
                a = np.zeros((1, self.dimension), dtype=np.float32)
                a[0, : (len(query) % (self.dimension - 1)) + 1] = 1.0
                return a
        emb = _MockEmbeddingEngine(dim=64)
        pipeline.vector_store = VectorStore(emb, str(tmpdir / f"vs_{session_id}"))
    # Override generator model if requested (doesn't mutate registry permanently)
    if model_name:
        try:
            pipeline.generator = build_generator(use_mock=use_mock, model=model_name)
        except Exception:
            logger.warning("Failed to override generator with model %s", model_name, exc_info=True)
    else:
        pipeline.vector_store = VectorStore(
            registry.embedding_engine,
            str(tmpdir / f"vs_{session_id}")
        )
    return pipeline


# ─────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/health")
def health():
    """Check if Ollama/Mistral is reachable."""
    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        mistral_ready = any("mistral" in m for m in models)
        return jsonify({
            "status": "ok",
            "ollama": True,
            "mistral": mistral_ready,
            "models": models
        })
    except Exception:
        return jsonify({"status": "ok", "ollama": False, "mistral": False, "models": []})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Accept .txt or .pdf, return extracted plain text."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = file.filename.lower()
    content = ""

    try:
        if filename.endswith(".txt"):
            content = file.read().decode("utf-8", errors="replace")

        elif filename.endswith(".pdf"):
            # Try pdfplumber first, then fallback to pypdf
            file_bytes = file.read()
            tmp_path = UPLOAD_FOLDER / f"{uuid.uuid4()}.pdf"
            tmp_path.write_bytes(file_bytes)
            try:
                import pdfplumber
                with pdfplumber.open(str(tmp_path)) as pdf:
                    content = "\n\n".join(
                        page.extract_text() or "" for page in pdf.pages
                    )
            except ImportError:
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(str(tmp_path))
                    content = "\n\n".join(
                        page.extract_text() or "" for page in reader.pages
                    )
                except ImportError:
                    return jsonify({"error": "PDF support not installed. Run: pip install pdfplumber"}), 500
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            return jsonify({"error": "Only .txt and .pdf files are supported"}), 400

        content = content.strip()
        if not content:
            return jsonify({"error": "Could not extract text from file"}), 400

        return jsonify({
            "text": content,
            "chars": len(content),
            "filename": file.filename
        })

    except Exception as e:
        logger.error("Upload error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
def ask():
    """
    Body (JSON):
    {
        "question": "...",
        "documents": [
            {"id": "doc1", "text": "..."},
            ...
        ]
    }
    Returns:
    {
        "answer": "...",
        "sub_queries": [...],
        "evidence": [
            {
                "rank": 1,
                "source": "doc1",
                "text": "...",
                "scores": {
                    "similarity": 0.9,
                    "reranker": 0.8,
                    "density": 0.6,
                    "total": 0.85
                }
            }
        ],
        "latency_ms": 1234
    }
    """
    import time

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    question = (data.get("question") or "").strip()
    documents = data.get("documents") or []

    if not question:
        return jsonify({"error": "Question is required"}), 400
    if not documents:
        return jsonify({"error": "At least one document is required"}), 400

    # Validate documents
    clean_docs = []
    for i, doc in enumerate(documents):
        text = (doc.get("text") or "").strip()
        doc_id = (doc.get("id") or f"doc_{i+1}").strip()
        if text:
            clean_docs.append({"id": doc_id, "text": text})

    if not clean_docs:
        return jsonify({"error": "All documents are empty"}), 400

    session_id = str(uuid.uuid4())[:8]
    t0 = time.time()

    try:
        use_mock = bool(data.get("mock", False))
        pipeline = build_pipeline(session_id, use_mock=use_mock)

        # Ingest
        pipeline.ingest(clean_docs)

        # Get sub-queries for display
        sub_queries = pipeline.decomposer.decompose(question)

        # Get answer + evidence
        answer, evidence_items = pipeline.answer_with_evidence(question)

        latency_ms = int((time.time() - t0) * 1000)

        evidence_out = []
        for item in evidence_items:
            # Truncate text for display
            text_preview = item.text[:400] + ("..." if len(item.text) > 400 else "")
            evidence_out.append({
                "rank": item.evidence_rank,
                "source": item.source_doc_id,
                "text": text_preview,
                "scores": {
                    "similarity": round(item.similarity_score, 3),
                    "reranker":   round(item.reranker_score,   3),
                    "density":    round(item.density_score,    3),
                    "total":      round(item.evidence_score,   3),
                }
            })

        return jsonify({
            "answer":      answer,
            "sub_queries": sub_queries,
            "evidence":    evidence_out,
            "latency_ms":  latency_ms,
            "num_docs":    len(clean_docs),
        })

    except Exception as e:
        logger.error("Pipeline error: %s", traceback.format_exc())
        return jsonify({"error": f"Pipeline error: {str(e)}"}), 500


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*52)
    print("  WEAR-RAG Web App")
    print("  http://localhost:5000")
    print("="*52 + "\n")
    app.run(debug=False, port=5000, threaded=False)
