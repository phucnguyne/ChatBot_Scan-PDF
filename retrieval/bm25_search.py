import pickle
from config import BM25_INDEX_PATH

_bm25_index = None

def _load():
    global _bm25_index
    if _bm25_index is None:
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_index = pickle.load(f)
    return _bm25_index

def bm25_search(query: str, top_k: int = 4):
    idx = _load()
    tokens = query.split()
    scores = []
    N, avgdl = len(idx["docs"]), idx["avgdl"]
    k1, b    = idx["k1"], idx["b"]
    for i, (tf, length) in enumerate(zip(idx["doc_token_freqs"], idx["doc_lengths"])):
        score = 0.0
        for t in tokens:
            if t not in idx["idf"]:
                continue
            f   = tf.get(t, 0)
            idf = idx["idf"][t]
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * length / avgdl))
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [
        {"content": idx["docs"][i], "metadata": idx["metadatas"][i], "score": s}
        for i, s in scores[:top_k]
    ]