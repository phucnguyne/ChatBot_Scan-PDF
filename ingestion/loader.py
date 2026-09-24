from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path: str):
    """Load PDF, trả về list Document."""
    documents = PyPDFLoader(pdf_path).load()
    for document in documents:
        page = document.metadata.get("page")
        if isinstance(page, int):
            document.metadata["page"] = page + 1
    return documents

def document_has_text(documents) -> bool:
    return any((getattr(d, "page_content", "") or "").strip() for d in documents)