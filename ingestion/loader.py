from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path: str):
    """Load PDF, trả về list Document."""
    return PyPDFLoader(pdf_path).load()

def document_has_text(documents) -> bool:
    return any((getattr(d, "page_content", "") or "").strip() for d in documents)