import os
import json
import pickle
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"
bm25_index_path = os.path.join(vector_db_path, "bm25_index.pkl")


def _ensure_basename(metadata: dict) -> dict:
    """Normalize metadata so we store short source file names."""
    md = dict(metadata or {})
    src = md.get("source") or md.get("file") or md.get("path")
    if src:
        md["source"] = os.path.basename(src)
    return md


def create_db_from_files(chunk_size: int = 512, chunk_overlap: int = 50, build_bm25: bool = True):
    # Khai bao loader de quet toan bo thu muc data
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise ValueError("No PDF files found or loaded from 'data/'. Add PDFs to data/ and try again.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks created from PDFs. Check that PDFs contain extractable text.")

    # Ensure each chunk has useful metadata for citation and filtering
    for i, ch in enumerate(chunks):
        md = _ensure_basename(ch.metadata)
        # preserve page if provided by loader
        if "page" in ch.metadata:
            md["page"] = ch.metadata.get("page")
        # optional fields for later use
        md.setdefault("chunk_id", f"chunk_{i}")
        md.setdefault("source", md.get("source", "unknown"))
        ch.metadata = md

    # Embedding
    embedding_model_file = os.environ.get("EMBEDDING_MODEL_FILE", "models/all-MiniLM-L6-v2-f16.gguf")
    embedding_model = GPT4AllEmbeddings(model_file=embedding_model_file)
    db = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(vector_db_path, exist_ok=True)
    db.save_local(vector_db_path)

    # Optional: build BM25 index for hybrid search
    if build_bm25 and BM25Okapi is not None:
        try:
            tokenized_texts: List[List[str]] = []
            metadatas: List[dict] = []
            for ch in chunks:
                # simple whitespace tokenization; rank_bm25 expects token lists
                tokenized = ch.page_content.split()
                tokenized_texts.append(tokenized)
                metadatas.append(ch.metadata)
            bm25 = BM25Okapi(tokenized_texts)
            with open(bm25_index_path, "wb") as fh:
                pickle.dump({"bm25": bm25, "metadatas": metadatas, "docs": [c.page_content for c in chunks]}, fh)
        except Exception:
            # don't fail the whole run on BM25 build errors
            pass

    return db


if __name__ == "__main__":
    os.makedirs(vector_db_path, exist_ok=True)
    create_db_from_files()