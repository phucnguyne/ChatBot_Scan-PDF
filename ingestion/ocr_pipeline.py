import gc
import os
import tempfile

from langchain_core.documents import Document


def ocr_pdf(pdf_path: str, start_page: int = 1, page_count: int = 0):
    """OCR selected pages of a PDF while keeping peak memory low."""
    try:
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        from pdf2image import convert_from_path
        from pypdf import PdfReader
        from paddleocr import PaddleOCR
    except ImportError:
        return []

    try:
        ocr = PaddleOCR(
            lang=os.environ.get("OCR_LANG", "en"),
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        ocr = PaddleOCR(use_angle_cls=False, lang=os.environ.get("OCR_LANG", "en"))

    total_pages = len(PdfReader(pdf_path).pages)
    start_page = max(1, start_page)
    end_page = total_pages if page_count <= 0 else min(total_pages, start_page + page_count - 1)
    docs = []
    with tempfile.TemporaryDirectory() as tmp:
        for page_number in range(start_page, end_page + 1):
            images = convert_from_path(
                pdf_path, dpi=120, first_page=page_number,
                last_page=page_number, fmt="png",
            )
            if not images:
                continue
            image = images[0]
            image_path = os.path.join(tmp, f"page_{page_number}.png")
            image.save(image_path, "PNG")
            if hasattr(ocr, "predict"):
                result = list(ocr.predict(image_path))
            else:
                result = ocr.ocr(image_path, cls=False) or []
            docs.append(Document(
                page_content="\n".join(_extract_lines(result)),
                metadata={"source": os.path.basename(pdf_path), "page": page_number},
            ))
            image.close()
            del images, image, result
            gc.collect()
    return docs


def _extract_lines(result):
    lines = []
    for block in result:
        if isinstance(block, dict):
            texts = block.get("rec_texts") or []
            lines.extend(text.strip() for text in texts if isinstance(text, str) and text.strip())
            continue
        try:
            texts = block["rec_texts"]
        except (KeyError, TypeError, IndexError):
            texts = []
        if texts:
            lines.extend(text.strip() for text in texts if isinstance(text, str) and text.strip())
            continue
        if isinstance(block, list):
            for line in block:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    content = line[1]
                    if isinstance(content, (list, tuple)):
                        content = content[0]
                    if isinstance(content, str) and content.strip():
                        lines.append(content.strip())
    return lines
