"""
build_index.py — Chạy một lần để build FAISS + BM25 index từ PDF trong data/pdfs/
Usage: python build_index.py
"""
from pathlib import Path
from ingestion.loader import load_pdf, document_has_text
from ingestion.ocr_pipeline import ocr_pdf
from ingestion.chunker import split_documents
from ingestion.metadata import enrich_metadata
from ingestion.index_builder import build_bm25, save_bm25, build_faiss

docs = []
pdf_files = list(Path("data/pdfs").glob("*.pdf"))

if not pdf_files:
    print("⚠️  Không tìm thấy PDF nào trong data/pdfs/")
else:
    for p in pdf_files:
        print(f"📄 Đang load: {p.name}")
        raw = load_pdf(str(p))
        docs += raw if document_has_text(raw) else ocr_pdf(str(p))

    chunks = enrich_metadata(split_documents(docs))
    print(f"✅ Tổng chunks: {len(chunks)}")

    save_bm25(build_bm25(chunks))
    build_faiss(chunks)
    print("🎉 Done! Chunks:", len(chunks))