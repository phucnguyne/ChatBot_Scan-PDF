import os
import json
import pickle
import math
import tempfile
from pathlib import Path
from collections import Counter
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

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


def _document_has_text(documents) -> bool:
    return any((getattr(doc, "page_content", "") or "").strip() for doc in documents)


def _ocr_pdf_file(pdf_path: str):
    if convert_from_path is None or PaddleOCR is None:
        return []

    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="vi")
    except Exception:
        return []

    ocr_documents = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            images = convert_from_path(pdf_path, dpi=200)
            for page_number, image in enumerate(images, start=1):
                image_path = os.path.join(tmpdir, f"page_{page_number}.png")
                image.save(image_path, format="PNG")
                try:
                    result = ocr.ocr(image_path, cls=True)
                except Exception:
                    result = []

                lines = []
                for block in result or []:
                    if isinstance(block, list):
                        for line in block:
                            if isinstance(line, (list, tuple)) and len(line) >= 2:
                                content = line[1]
                                if isinstance(content, (list, tuple)) and content:
                                    content = content[0]
                                if isinstance(content, str) and content.strip():
                                    lines.append(content.strip())
                    elif isinstance(block, (list, tuple)) and len(block) >= 2:
                        content = block[1]
                        if isinstance(content, (list, tuple)) and content:
                            content = content[0]
                        if isinstance(content, str) and content.strip():
                            lines.append(content.strip())

                ocr_documents.append(
                    Document(
                        page_content="\n".join(lines).strip(),
                        metadata={"source": os.path.basename(pdf_path), "page": page_number},
                    )
                )
    except Exception:
        return []

    return ocr_documents


def _load_pdf_documents(pdf_path: str):
    documents = PyPDFLoader(pdf_path).load()
    if _document_has_text(documents):
        return documents

    ocr_documents = _ocr_pdf_file(pdf_path)
    return ocr_documents or documents


def _build_bm25_index(chunks: List):
    tokenized_docs = [chunk.page_content.split() for chunk in chunks]
    doc_token_freqs = []
    doc_freqs = Counter()
    doc_lengths = []

    for tokens in tokenized_docs:
        token_freq = Counter(tokens)
        doc_token_freqs.append(dict(token_freq))
        doc_lengths.append(len(tokens))
        for token in token_freq:
            doc_freqs[token] += 1

    total_docs = len(tokenized_docs)
    avgdl = (sum(doc_lengths) / total_docs) if total_docs else 0.0
    idf = {
        token: math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
        for token, df in doc_freqs.items()
    }

    return {
        "tokenized_docs": tokenized_docs,
        "doc_token_freqs": doc_token_freqs,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "idf": idf,
        "docs": [chunk.page_content for chunk in chunks],
        "metadatas": [chunk.metadata for chunk in chunks],
        "k1": 1.5,
        "b": 0.75,
    }


def create_db_from_files(chunk_size: int = 512, chunk_overlap: int = 50, build_bm25: bool = True):
    pdf_paths = sorted(Path(pdf_data_path).glob("*.pdf"))
    if not pdf_paths:
        raise ValueError("No PDF files found or loaded from 'data/'. Add PDFs to data/ and try again.")

    documents = []
    for pdf_path in pdf_paths:
        documents.extend(_load_pdf_documents(str(pdf_path)))

    if not documents:
        raise ValueError("No PDF documents could be loaded or OCR'd from 'data/'.")

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

    # Build the BM25 artifact first so hybrid search metadata is available even if
    # the embedding model load or FAISS write is slow.
    if build_bm25:
        try:
            bm25_index = _build_bm25_index(chunks)
            with open(bm25_index_path, "wb") as fh:
                pickle.dump(bm25_index, fh)
        except Exception:
            # don't fail the whole run on BM25 build errors
            pass

    # Embedding
    embedding_model_file = os.environ.get("EMBEDDING_MODEL_FILE", "models/all-MiniLM-L6-v2-f16.gguf")
    embedding_model = GPT4AllEmbeddings(model_file=embedding_model_file)
    db = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(vector_db_path, exist_ok=True)
    db.save_local(vector_db_path)

    return db


if __name__ == "__main__":
    os.makedirs(vector_db_path, exist_ok=True)
    create_db_from_files()