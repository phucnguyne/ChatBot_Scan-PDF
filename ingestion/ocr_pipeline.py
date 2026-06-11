import os, tempfile
from langchain_core.documents import Document

def ocr_pdf(pdf_path: str):
    """OCR một file PDF, trả về list Document."""
    try:
        from pdf2image import convert_from_path
        from paddleocr import PaddleOCR
    except ImportError:
        return []

    ocr = PaddleOCR(use_angle_cls=True, lang="vi")
    docs = []
    with tempfile.TemporaryDirectory() as tmp:
        images = convert_from_path(pdf_path, dpi=200)
        for i, img in enumerate(images, 1):
            p = os.path.join(tmp, f"page_{i}.png")
            img.save(p, "PNG")
            result = ocr.ocr(p, cls=True) or []
            lines = _extract_lines(result)
            docs.append(Document(
                page_content="\n".join(lines),
                metadata={"source": os.path.basename(pdf_path), "page": i},
            ))
    return docs

def _extract_lines(result):
    lines = []
    for block in result:
        if isinstance(block, list):
            for line in block:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    content = line[1]
                    if isinstance(content, (list, tuple)):
                        content = content[0]
                    if isinstance(content, str) and content.strip():
                        lines.append(content.strip())
    return lines