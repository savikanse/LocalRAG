from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT/ "data"

PAPERS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"

PAPERS_JSON = PAPERS_DIR / "papers.json"
INDEX_DIR = DATA_DIR / "index"

FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
BM25_PATH = INDEX_DIR / "bm25.pkl"


# ============================================================
# OLLAMA
# ============================================================

OLLAMA_HOST = "http://localhost:11434"

LLM_MODEL = "llama3.2:3b"

EMBED_MODEL = "embeddinggemma"

JUDGE_MODEL="qwen3:8b"
# ============================================================
# RERANKER
# ============================================================

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE = 800

CHUNK_OVERLAP = 120


# ============================================================
# RETRIEVAL
# ============================================================

DENSE_K = 12

BM25_K = 12

RRF_K = 60
RERANK_CANDIDATES = 15

RERANK_K = 5