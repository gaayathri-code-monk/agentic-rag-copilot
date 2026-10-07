"""
config.py
Single source of truth for all pipeline settings.
Swap values here — nothing else needs touching.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "pdfs"          # drop your PDFs here
CHROMA_DIR  = ROOT / "data" / "chroma_db"     # persisted vector store

# ── Embedding model (local, no API key) ───────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
# EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # alternative: slightly better accuracy

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512    # tokens per chunk
CHUNK_OVERLAP = 64     # overlap to preserve context across boundaries

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "rag_copilot_papers"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K          = 5     # number of chunks retrieved per query
RETRIEVAL_SCORE_THRESHOLD = 0.35   # cosine similarity min — below this = weak retrieval

# ── LLM (local Ollama — swap to OpenAI if needed) ─────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")   # "ollama" | "openai"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
