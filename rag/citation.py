import os

def build_citation_string(source_docs) -> str:
    """Trả về chuỗi '📌 Nguồn: file.pdf tr.3 | file2.pdf tr.5'"""
    seen, sources = set(), []
    for d in source_docs:
        md    = getattr(d, "metadata", {}) or {}
        src   = os.path.basename(md.get("source") or "?")
        page  = md.get("page")
        label = f"{src} tr.{page}" if page is not None else src
        if label not in seen:
            sources.append(label)
            seen.add(label)
    return "📌 Nguồn: " + " | ".join(sources) if sources else ""