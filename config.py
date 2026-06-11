import os
from pathlib import Path

PROJECT_ROOT      = Path(__file__).resolve().parent
VECTOR_DB_PATH    = str(PROJECT_ROOT / "vectorstores" / "db_faiss")
BM25_INDEX_PATH   = os.path.join(VECTOR_DB_PATH, "bm25_index.pkl")
SESSION_MEMORY    = str(PROJECT_ROOT / "vectorstores" / "session_memory.jsonl")
CHAT_HISTORY_PATH = str(PROJECT_ROOT / "vectorstores" / "chat_history.jsonl")
EMBEDDING_MODEL   = str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2-f16.gguf")
RERANKER_MODEL    = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
POPPLER_PATH      = r"C:\poppler-26.02.0\Library\bin"   # Windows only

PDF_DATA_PATH     = str(PROJECT_ROOT / "data" / "pdfs")
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 50
TOP_K_RETRIEVAL   = 4
TOP_K_RERANK      = 2