# ingestion/index_builder.py
import os, pickle, math
from collections import Counter
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from config import VECTOR_DB_PATH, BM25_INDEX_PATH, EMBEDDING_MODEL


def build_bm25(chunks) -> dict:
    tokenized = [ch.page_content.split() for ch in chunks]
    freq_list, doc_freqs, lengths = [], Counter(), []
    for tokens in tokenized:
        tf = Counter(tokens)
        freq_list.append(dict(tf))
        lengths.append(len(tokens))
        for t in tf:
            doc_freqs[t] += 1
    N = len(tokenized)
    avgdl = sum(lengths) / N if N else 0
    idf = {t: math.log(1 + (N - df + 0.5) / (df + 0.5)) for t, df in doc_freqs.items()}
    return {
        "tokenized_docs": tokenized, "doc_token_freqs": freq_list,
        "doc_lengths": lengths, "avgdl": avgdl, "idf": idf,
        "docs": [ch.page_content for ch in chunks],
        "metadatas": [ch.metadata for ch in chunks],
        "k1": 1.5, "b": 0.75,
    }


def save_bm25(index: dict):
    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)


def build_faiss(chunks):
    emb = GPT4AllEmbeddings(model_file=EMBEDDING_MODEL)
    db  = FAISS.from_documents(chunks, emb)
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    db.save_local(VECTOR_DB_PATH)
    return db