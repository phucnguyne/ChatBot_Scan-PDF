"""Build the FAISS + BM25 index from imported PDFs."""

import argparse
import json
import os
from pathlib import Path
from typing import Callable

from ingestion.chunker import split_documents
from ingestion.index_builder import build_bm25, build_faiss, save_bm25
from ingestion.loader import document_has_text, load_pdf
from ingestion.metadata import enrich_metadata
from ingestion.ocr_pipeline import ocr_pdf
from config import INDEX_MANIFEST_PATH, PDF_DATA_PATH, BM25_INDEX_PATH, VECTOR_DB_PATH


def build_index(start_page=1, page_count=0, ocr_lang="en", progress: Callable | None = None,
                data_path=PDF_DATA_PATH, vector_db_path=VECTOR_DB_PATH,
                bm25_index_path=BM25_INDEX_PATH, manifest_path=INDEX_MANIFEST_PATH):
    if start_page < 1 or page_count < 0:
        raise ValueError("Trang bắt đầu phải >= 1 và số trang phải >= 0.")
    os.environ["OCR_LANG"] = ocr_lang
    pdf_files = list(Path(data_path).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Chưa có PDF trong {data_path}.")

    def report(payload):
        if progress:
            progress(payload)

    docs = []
    report({"phase": "loading", "message": f"Đang xử lý {len(pdf_files)} tài liệu..."})
    for number, path in enumerate(pdf_files, 1):
        report({"phase": "loading", "file": path.name, "file_number": number, "file_total": len(pdf_files)})
        raw = load_pdf(str(path))
        first = start_page - 1
        selected_raw = raw[first:] if page_count == 0 else raw[first:first + page_count]
        if document_has_text(selected_raw):
            docs += selected_raw
        else:
            docs += ocr_pdf(str(path), start_page=start_page, page_count=page_count)
        report({"phase": "page", "file": path.name, "file_number": number, "file_total": len(pdf_files)})

    chunks = enrich_metadata(split_documents(docs))
    report({"phase": "embedding", "message": f"Đang tạo embedding cho {len(chunks)} đoạn..."})
    save_bm25(build_bm25(chunks), bm25_index_path)
    build_faiss(chunks, vector_db_path)
    manifest = {
        "files": [
            {"name": path.name, "size": path.stat().st_size, "modified": path.stat().st_mtime_ns}
            for path in pdf_files
        ],
        "start_page": start_page,
        "page_count": page_count,
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"chunks": len(chunks), "files": len(pdf_files)}
    report({"phase": "done", **result})
    return result


def main():
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 index from PDFs")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--page-count", type=int, default=0)
    parser.add_argument("--max-pages", type=int, help="Alias của --page-count")
    parser.add_argument("--ocr-lang", default="en")
    args = parser.parse_args()
    count = args.max_pages if args.max_pages is not None else args.page_count
    result = build_index(args.start_page, count, args.ocr_lang, print)
    print(f"Done! Chunks: {result['chunks']}")


if __name__ == "__main__":
    main()
