# ingestion/metadata.py
import os

def enrich_metadata(chunks):
    """
    Assign a chunk_id and normalize the source basename for each chunk.
    Target metadata structure:
    {
      "id":          "chunk_001",
      "source":      "AI.pdf",
      "page":        12,
      "chunk_index": 5,
    }
    """
    for i, ch in enumerate(chunks):
        md = dict(ch.metadata or {})
        src = md.get("source") or md.get("file") or md.get("path")
        if src:
            md["source"] = os.path.basename(src)
        md.setdefault("chunk_id",    f"chunk_{i:04d}")
        md.setdefault("chunk_index", i)
        md.setdefault("source",      "unknown")
        ch.metadata = md
    return chunks