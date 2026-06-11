# ingestion/index_builder.py
import os, pickle, math
from pathlib import Path

from langchain_community.vectorstores import FAISS  # pyrefly: ignore[missing-import]
from langchain_core.embeddings import Embeddings  # pyrefly: ignore[missing-import]
from config import VECTOR_DB_PATH, BM25_INDEX_PATH, EMBEDDING_MODEL


# ── Sentence-Transformers embedding wrapper ───────────────────────────────────
class SentenceTransformerEmbedding(Embeddings):
    """
    Sử dụng sentence-transformers trực tiếp — native, nhanh, ổn định.
    Hỗ trợ cả HuggingFace model name (vd: 'all-MiniLM-L6-v2')
    lẫn đường dẫn local (vd: './models/my-model').
    """
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # pyrefly: ignore[missing-import]
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25(chunks) -> dict:
    from collections import Counter
    tokenized = [ch.page_content.split() for ch in chunks]
    freq_list, doc_freqs, lengths = [], Counter(), []
    for tokens in tokenized:
        tf = Counter(tokens)
        freq_list.append(dict(tf))
        lengths.append(len(tokens))
        for t in tf:
            doc_freqs[t] += 1
    N     = len(tokenized)
    avgdl = sum(lengths) / N if N else 0
    idf   = {
        t: math.log(1 + (N - df + 0.5) / (df + 0.5))
        for t, df in doc_freqs.items()
    }
    return {
        "tokenized_docs":  tokenized,
        "doc_token_freqs": freq_list,
        "doc_lengths":     lengths,
        "avgdl":           avgdl,
        "idf":             idf,
        "docs":            [ch.page_content for ch in chunks],
        "metadatas":       [ch.metadata for ch in chunks],
        "k1": 1.5,
        "b":  0.75,
    }


def save_bm25(index: dict):
    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    print(f"✅ BM25 saved → {BM25_INDEX_PATH}")


# ── FAISS ─────────────────────────────────────────────────────────────────────

def build_faiss(chunks):
    if not chunks:
        raise ValueError("Không có chunk nào để build FAISS.")

    print(f"🔧 Đang embed {len(chunks)} chunks với model: {EMBEDDING_MODEL}")

    emb = SentenceTransformerEmbedding(model_name=EMBEDDING_MODEL)

    # Kiểm tra nhanh embed 1 chunk trước khi xử lý toàn bộ
    test_vec = emb.embed_query(chunks[0].page_content[:200])
    if not test_vec:
        raise RuntimeError(
            "Embedding trả về vector rỗng. "
            "Kiểm tra lại model name hoặc thử model khác."
        )
    print(f"✅ Embedding OK — dim={len(test_vec)}")

    db = FAISS.from_documents(chunks, emb)
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    db.save_local(VECTOR_DB_PATH)
    print(f"✅ FAISS saved → {VECTOR_DB_PATH}  ({len(chunks)} chunks)")
    return db