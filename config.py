import os
from pathlib import Path

PROJECT_ROOT      = Path(__file__).resolve().parent
VECTOR_DB_PATH    = str(PROJECT_ROOT / "vectorstores" / "db_faiss")
BM25_INDEX_PATH   = os.path.join(VECTOR_DB_PATH, "bm25_index.pkl")
INDEX_MANIFEST_PATH = os.path.join(VECTOR_DB_PATH, "index_manifest.json")
SESSION_MEMORY    = str(PROJECT_ROOT / "vectorstores" / "session_memory.jsonl")
CHAT_HISTORY_PATH = str(PROJECT_ROOT / "vectorstores" / "chat_history.jsonl")
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
RERANKER_MODEL    = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
POPPLER_PATH      = r"C:\poppler-26.02.0\Library\bin"   # Windows only

PDF_DATA_PATH     = os.environ.get("PDF_DATA_PATH", str(PROJECT_ROOT / "documents"))
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 50
TOP_K_RETRIEVAL   = 4
TOP_K_RERANK      = 2
TOP_K_LONG_ANSWER = 8