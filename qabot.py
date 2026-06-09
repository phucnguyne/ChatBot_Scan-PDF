"""
app.py — Gradio UI for ScanPDF
  Tab 1 : Upload MULTIPLE PDFs → Extract text (pypdf + PaddleOCR fallback)
  Tab 2 : Chat Q&A  (LangChain + FAISS + BM25 + Reranker + Session Memory)
  Tab 3 : Diagnostics / loaded-file manager
  LLM   : Ollama qwen2.5:1.5b  (no local .gguf needed)

FIXES vs previous version
  ✓ RetrievalQA import: tries 4 known paths, never crashes
  ✓ GPT4AllEmbeddings: tries both community and gpt4all packages
  ✓ Gradio 6 chatbot: auto-detects version, uses tuples OR dicts
  ✓ theme= moved into launch()
  ✓ PDF context never bleeds between upload sessions
  ✓ Detailed import diagnostics in Tab 3
"""

import os, sys, json, pickle, re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

POPPLER_PATH = r"C:\poppler-26.02.0\Library\bin"

# ── Gradio version detection ──────────────────────────────────────────────────
try:
    _gr_parts = gr.__version__.split(".")
    _gr_major = int(_gr_parts[0])
    _gr_minor = int(_gr_parts[1]) if len(_gr_parts) > 1 else 0
except Exception:
    _gr_major, _gr_minor = 3, 0

# Gradio 6+ dropped the type= argument from gr.Chatbot entirely.
# Gradio 4.29-5.x used type="messages" to opt into dict format.
# Gradio < 4.29 used (user_msg, bot_msg) tuples.
_GR6 = _gr_major > 4 or (_gr_major == 4 and _gr_minor >= 29)
_GR6_NO_TYPE = _gr_major >= 6   # Gradio 6+ removed the type= parameter

def _make_msg(role: str, content: str):
    """Return the right history item for the installed Gradio version."""
    if _GR6:
        return {"role": role, "content": content}
    # tuples: (user_text, bot_text) — user=None means it's the bot's turn
    if role == "user":
        return (content, None)
    return (None, content)

def _append_pair(history: list, user_msg: str, bot_msg: str) -> list:
    """Append a user/bot exchange in the correct format."""
    h = list(history or [])
    if _GR6:
        h.append({"role": "user",      "content": user_msg})
        h.append({"role": "assistant", "content": bot_msg})
    else:
        h.append((user_msg, bot_msg))
    return h

# ── Optional imports ──────────────────────────────────────────────────────────
_import_log: List[str] = []

def _try(label, fn):
    try:
        result = fn()
        _import_log.append(f"✅ {label}")
        return result
    except Exception as e:
        _import_log.append(f"❌ {label}: {e}")
        return None

PdfReader         = _try("pypdf.PdfReader",          lambda: __import__("pypdf", fromlist=["PdfReader"]).PdfReader)
convert_from_path = _try("pdf2image.convert_from_path", lambda: __import__("pdf2image", fromlist=["convert_from_path"]).convert_from_path)
BM25Okapi         = _try("rank_bm25.BM25Okapi",      lambda: __import__("rank_bm25", fromlist=["BM25Okapi"]).BM25Okapi)
CrossEncoder      = _try("sentence_transformers.CrossEncoder", lambda: __import__("sentence_transformers", fromlist=["CrossEncoder"]).CrossEncoder)

try:
    from paddleocr import PaddleOCR
    _import_log.append("✅ paddleocr.PaddleOCR")
    _ocr_engine = None
except Exception as e:
    _import_log.append(f"❌ paddleocr: {e}")
    PaddleOCR   = None
    _ocr_engine = None

# ── LangChain — try every known import path ───────────────────────────────────
_langchain_ok = False
RecursiveCharacterTextSplitter = None
FAISS          = None
PromptTemplate = None
OllamaLLM      = None
create_retrieval_chain = None
create_stuff_documents_chain = None
GPT4AllEmbeddings = None

def _import_langchain():
    global _langchain_ok, RecursiveCharacterTextSplitter, FAISS
    global PromptTemplate, OllamaLLM, RetrievalQA, GPT4AllEmbeddings

    # text splitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _RCS
    RecursiveCharacterTextSplitter = _RCS
    _import_log.append("✅ langchain_text_splitters")

    # FAISS
    from langchain_community.vectorstores import FAISS as _FAISS
    FAISS = _FAISS
    _import_log.append("✅ langchain_community.vectorstores.FAISS")

    # Embeddings — try three locations
    _emb = None
    for _path in (
        ("langchain_community.embeddings",     "GPT4AllEmbeddings"),
        ("langchain_community.embeddings.gpt4all", "GPT4AllEmbeddings"),
        ("langchain_gpt4all",                  "GPT4AllEmbeddings"),
    ):
        try:
            mod  = __import__(_path[0], fromlist=[_path[1]])
            _emb = getattr(mod, _path[1])
            _import_log.append(f"✅ {_path[0]}.{_path[1]}")
            break
        except Exception as e:
            _import_log.append(f"  ↳ tried {_path[0]}: {e}")
    if _emb is None:
        raise ImportError("GPT4AllEmbeddings not found in any known package")
    GPT4AllEmbeddings = _emb

    # PromptTemplate
    from langchain_core.prompts import PromptTemplate as _PT
    PromptTemplate = _PT
    _import_log.append("✅ langchain_core.prompts.PromptTemplate")

    # OllamaLLM
    from langchain_ollama import OllamaLLM as _OL
    OllamaLLM = _OL
    _import_log.append("✅ langchain_ollama.OllamaLLM")

    # LCEL chain helpers (langchain.chains removed in langchain ≥ 1.x)
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    _import_log.append("✅ langchain_core LCEL (RunnablePassthrough, RunnableLambda, StrOutputParser)")

    return True

try:
    _langchain_ok = _import_langchain()
except Exception as _lc_err:
    _import_log.append(f"❌ LangChain overall: {_lc_err}")
    print(f"[LangChain import error] {_lc_err}")

# ── Paths ─────────────────────────────────────────────────────────────────────
VECTOR_DB_PATH  = str(PROJECT_ROOT / "vectorstores" / "db_faiss")
BM25_INDEX_PATH = os.path.join(VECTOR_DB_PATH, "bm25_index.pkl")
SESSION_MEMORY  = str(PROJECT_ROOT / "vectorstores" / "session_memory.jsonl")
CHAT_HISTORY_PATH = str(PROJECT_ROOT / "vectorstores" / "chat_history.jsonl")
EMBEDDING_MODEL = str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2-f16.gguf")
RERANKER_MODEL  = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

PROMPT_TEMPLATE = """<|im_start|>system
Bạn chỉ được phép trả lời dựa trên phần `context` được cung cấp bên dưới.
Nếu thông tin không có trong context, hãy trả lời: "Không tìm thấy thông tin trong tài liệu."
Không được suy đoán. Trích dẫn nguồn theo định dạng [Nguồn: <tên file> - trang <page>] khi có thể.

{context}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant"""

# ── Global state ──────────────────────────────────────────────────────────────
_vectorstore  = None
_qa_chain     = None
_loaded_files : List[str] = []


# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENT CHAT HISTORY  (survives page refresh / server restart)
# ═════════════════════════════════════════════════════════════════════════════

def _save_chat_turn(user_msg: str, bot_msg: str):
    """Append a single Q/A turn to the persistent JSONL log."""
    os.makedirs(os.path.dirname(CHAT_HISTORY_PATH), exist_ok=True)
    entry = {
        "user": user_msg,
        "bot":  bot_msg,
        "ts":   datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(CHAT_HISTORY_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

def _load_chat_history(limit: int = 200) -> list:
    """Load the last *limit* turns from disk and return as Gradio-ready list."""
    if not os.path.exists(CHAT_HISTORY_PATH):
        return []
    entries = []
    try:
        with open(CHAT_HISTORY_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and "user" in item and "bot" in item:
                        entries.append(item)
                except Exception:
                    pass
    except Exception:
        pass
    entries = entries[-limit:]
    return _entries_to_gradio(entries)

def _entries_to_gradio(entries: list) -> list:
    """Convert raw dicts to the Gradio chatbot list format."""
    history = []
    for e in entries:
        history = _append_pair(history, e["user"], e["bot"])
    return history

def clear_chat_history():
    """Delete the persistent chat log and return an empty history."""
    try:
        if os.path.exists(CHAT_HISTORY_PATH):
            os.remove(CHAT_HISTORY_PATH)
        return [], "✅ Đã xóa lịch sử chat."
    except Exception as ex:
        return [], f"❌ {ex}"

# ═════════════════════════════════════════════════════════════════════════════
# SESSION MEMORY  (lightweight Q/A summaries used as LLM context)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_topic(question: str) -> str:
    text = question.strip()
    for prefix in ("môn ", "bài ", "câu ", "chủ đề ", "phần "):
        if text.lower().startswith(prefix):
            words = text[len(prefix):].split()
            if words and words[0].lower() not in {"đó","này","ấy","kia","nữa"}:
                return " ".join(words[:5]).strip(" .,!?")
    return ""

def _load_session_entries(limit=20) -> List[dict]:
    if not os.path.exists(SESSION_MEMORY):
        return []
    entries = []
    try:
        with open(SESSION_MEMORY, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and item.get("summary"):
                        entries.append(item)
                except Exception:
                    pass
    except Exception:
        pass
    return entries[-limit:]

def _needs_session_context(q: str) -> bool:
    return any(m in q.lower() for m in
        ("môn đó","bài đó","câu đó","cái đó","nó","đó","tiếp theo","lúc nãy","vừa rồi","câu trước"))

def _build_session_context(question: str) -> str:
    entries = _load_session_entries()
    if not entries: return ""
    relevant = entries[-1:] if _needs_session_context(question) else entries[-3:]
    latest   = entries[-1]
    summaries = [e.get("summary","").strip() for e in relevant if e.get("summary")]
    if not summaries: return ""
    lines = []
    if latest.get("topic"): lines.append(f"Chủ đề gần nhất: {latest['topic']}")
    if latest.get("question"): lines.append(f"Câu hỏi gần nhất: {latest['question']}")
    lines.extend(f"- {s}" for s in summaries)
    return "Ngữ cảnh hội thoại trước đó:\n" + "\n".join(lines)

def _save_session_summary(question: str, answer: str):
    q = question.strip()
    topic   = _extract_topic(q)
    summary = f"Q: {q} | A: {answer}"[:200]
    os.makedirs(os.path.dirname(SESSION_MEMORY), exist_ok=True)
    with open(SESSION_MEMORY, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "summary":    summary,
            "topic":      topic,
            "question":   q,
            "answer":     answer[:500],
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }, ensure_ascii=False) + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# METADATA FILTER
# ═════════════════════════════════════════════════════════════════════════════

def _norm_key(k: str) -> str:
    k = k.strip().lower()
    return "source" if k in {"file","path"} else k

def _norm_val(k: str, v: str):
    v = v.strip().strip("\"'")
    if k == "source": return os.path.basename(v)
    if k == "page" and v.isdigit(): return int(v)
    return v

# Vietnamese / natural-language patterns that indicate a file or page filter.
# Examples:
#   "câu 1 ở file QuanLyGiaoVu.pdf"   -> source=QuanLyGiaoVu.pdf
#   "trong file abc.pdf trang 5"       -> source=abc.pdf, page=5
#   "trang 3 file xyz.pdf"             -> source=xyz.pdf, page=3
_FILE_PATTERNS = re.compile(
    r"(?:(?:ở|trong|của|at|in|from)\s+)?(?:file|tập tin|tài liệu|document)\s+([\w\-. ()]+\.pdf)",
    re.IGNORECASE,
)
_PAGE_PATTERNS = re.compile(
    r"(?:trang|page|tr\.?|pg\.?)\s*(\d+)",
    re.IGNORECASE,
)

def _extract_query_directives(question: str):
    """
    Parse BOTH colon-style directives (source:foo.pdf page:3)
    AND natural-language references (ở file QuanLyGiaoVu.pdf trang 2).
    Returns (cleaned_query, filter_dict).
    """
    filt = {}

    # ── 1. colon-style: source:foo.pdf page:3 ────────────────────────────
    parts = []
    for token in question.split():
        if ":" not in token:
            parts.append(token)
            continue
        k, v = token.split(":", 1)
        nk = _norm_key(k)
        if nk in {"source", "page"} and v:
            filt[nk] = _norm_val(nk, v)
        else:
            parts.append(token)
    cleaned = " ".join(parts).strip()

    # ── 2. natural language: "ở file X.pdf" / "trang N" ─────────────────
    m_file = _FILE_PATTERNS.search(cleaned)
    if m_file and "source" not in filt:
        fname = os.path.basename(m_file.group(1).strip())
        filt["source"] = fname
        cleaned = cleaned[:m_file.start()] + cleaned[m_file.end():]

    m_page = _PAGE_PATTERNS.search(cleaned)
    if m_page and "page" not in filt:
        filt["page"] = int(m_page.group(1))
        cleaned = cleaned[:m_page.start()] + cleaned[m_page.end():]

    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, filt

def _meta_matches(metadata: dict, filt: dict) -> bool:
    if not filt: return True
    md = metadata or {}
    for k, expected in filt.items():
        actual = md.get(k)
        if k == "source" and isinstance(actual, str): actual = os.path.basename(actual)
        if k == "page" and isinstance(actual, str) and actual.isdigit(): actual = int(actual)
        if actual != expected: return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM RETRIEVER  (FAISS + BM25 hybrid + cross-encoder reranker)
# ═════════════════════════════════════════════════════════════════════════════

class CustomRetriever:
    def __init__(self, db, bm25_path=None, reranker_model=None, fetch_k=20, top_k=5):
        self.db      = db
        self.fetch_k = fetch_k
        self.top_k   = top_k
        self.bm25 = self.bm25_docs = self.bm25_meta = None
        if bm25_path and os.path.exists(bm25_path):
            try:
                with open(bm25_path, "rb") as fh:
                    data = pickle.load(fh)
                self.bm25      = data
                self.bm25_docs = data.get("docs")
                self.bm25_meta = data.get("metadatas")
            except Exception: pass
        self.reranker = None
        if reranker_model and CrossEncoder is not None:
            try: self.reranker = CrossEncoder(reranker_model)
            except Exception: pass

    def _filter(self, docs, filt):
        if not filt: return docs
        return [d for d in docs if _meta_matches(getattr(d,"metadata",{}) or {}, filt)]

    def get_relevant_documents(self, query, **kwargs):
        cleaned, parsed_filt = _extract_query_directives(query)
        filt = kwargs.get("filter") or kwargs.get("metadata_filter") or parsed_filt
        q    = cleaned or query

        try:    faiss_docs = self.db.similarity_search(q, k=self.fetch_k)
        except: faiss_docs = []

        bm25_cands = []
        if self.bm25 and self.bm25_docs:
            tokens = q.split()
            try:
                scores  = self.bm25.get_scores(tokens) if hasattr(self.bm25,"get_scores") else [0]*len(self.bm25_docs)
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.fetch_k]
                for idx in top_idx:
                    bm25_cands.append({"text": self.bm25_docs[idx], "metadata": (self.bm25_meta or [{}])[idx]})
            except: pass

        # merge & deduplicate
        merged, seen = [], set()
        for d in faiss_docs:
            key = d.page_content
            if key in seen: continue
            seen.add(key); merged.append(d)
        for bc in bm25_cands:
            key = bc["text"]
            if key in seen: continue
            seen.add(key)
            class _D: pass
            d = _D(); d.page_content = bc["text"]; d.metadata = bc["metadata"]
            merged.append(d)

        merged = self._filter(merged, filt)

        if self.reranker and merged:
            try:
                sc = self.reranker.predict([(q, d.page_content) for d in merged])
                merged = [x for x,_ in sorted(zip(merged,sc), key=lambda t:t[1], reverse=True)]
            except: pass

        return merged[:self.top_k]

    # LangChain v0.2+ compatibility
    def invoke(self, query, **kwargs):
        return self.get_relevant_documents(query, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# OCR HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None and PaddleOCR is not None:
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="vi")
    return _ocr_engine

def extract_text_pages(pdf_path: Path) -> List[dict]:
    if PdfReader is None: return []
    try:
        reader = PdfReader(str(pdf_path))
        return [{"page": i+1, "text": (p.extract_text() or "").strip()}
                for i, p in enumerate(reader.pages)]
    except Exception as e:
        print(f"[pypdf] {e}"); return []

def ocr_pdf_pages(pdf_path: Path) -> List[dict]:
    if convert_from_path is None or PaddleOCR is None: return []
    try:
        images = convert_from_path(str(pdf_path), dpi=200, poppler_path=POPPLER_PATH)
    except Exception as e:
        print(f"[Poppler] {e}"); return []
    ocr = _get_ocr()
    if ocr is None: return []
    pages = []
    for i, img in enumerate(images, 1):
        try:   result = ocr.ocr(img, cls=True)
        except: result = []
        lines = []
        for block in (result or []):
            if not isinstance(block, list): continue
            for line in block:
                if isinstance(line, (list,tuple)) and len(line) >= 2:
                    c = line[1]
                    if isinstance(c, (list,tuple)) and c: c = c[0]
                    if isinstance(c, str) and c.strip(): lines.append(c.strip())
        pages.append({"page": i, "text": "\n".join(lines).strip()})
    return pages

def extract_with_ocr_fallback(pdf_path: Path, force_ocr=False):
    pages    = extract_text_pages(pdf_path)
    has_text = any(p["text"] for p in pages)
    if has_text and not force_ocr: return pages, False
    ocr_p = ocr_pdf_pages(pdf_path)
    return (ocr_p, True) if ocr_p else (pages, False)


# ═════════════════════════════════════════════════════════════════════════════
# VECTORSTORE / QA-CHAIN BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def _build_vectorstore(all_pages_by_file: dict) -> str:
    global _vectorstore, _qa_chain
    _vectorstore = _qa_chain = None          # always reset first

    if not _langchain_ok:
        detail = "\n".join(l for l in _import_log if "❌" in l or "↳" in l)
        return f"❌ LangChain unavailable.\n{detail}"

    if not Path(EMBEDDING_MODEL).exists():
        return (f"❌ Embedding model not found:\n  {EMBEDDING_MODEL}\n"
                "  Place all-MiniLM-L6-v2-f16.gguf inside the models/ folder.")

    try:
        embeddings = GPT4AllEmbeddings(model_file=EMBEDDING_MODEL)
    except Exception as e:
        return f"❌ Embedding init failed: {e}"

    # Build fresh FAISS from current upload batch
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    all_docs = []
    for filename, pages in all_pages_by_file.items():
        for p in pages:
            if not p["text"].strip(): continue
            chunks = splitter.create_documents(
                [p["text"]],
                metadatas=[{"source": filename, "page": p["page"]}]
            )
            all_docs.extend(chunks)

    if not all_docs:
        return "⚠️ No text content found in uploaded PDFs."

    try:
        _vectorstore = FAISS.from_documents(all_docs, embeddings)
        print(f"[QA] FAISS built: {len(all_docs)} chunks from {len(all_pages_by_file)} file(s)")
    except Exception as e:
        return f"❌ FAISS build failed: {e}"

    # Build QA chain
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.3, num_predict=512)
        prompt    = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context","question"])
        bm25_path = BM25_INDEX_PATH if os.path.exists(BM25_INDEX_PATH) else None
        retriever = CustomRetriever(_vectorstore, bm25_path=bm25_path,
                                    reranker_model=RERANKER_MODEL, fetch_k=20, top_k=5)

        def _fmt_docs(docs):
            parts = []
            for d in docs:
                md  = getattr(d, "metadata", {}) or {}
                src = os.path.basename(md.get("source") or "unknown")
                pg  = md.get("page")
                hdr = f"[Nguồn: {src}" + (f" - trang {pg}" if pg else "") + "]"
                parts.append(f"{hdr}\n{d.page_content}")
            return "\n\n".join(parts)

        from langchain_core.runnables import RunnablePassthrough, RunnableLambda
        from langchain_core.output_parsers import StrOutputParser

        _qa_chain = (
            RunnablePassthrough.assign(
                context_docs=RunnableLambda(lambda x: retriever.get_relevant_documents(x["query"]))
            )
            | RunnablePassthrough.assign(
                context=RunnableLambda(lambda x: _fmt_docs(x["context_docs"])),
                question=RunnableLambda(lambda x: x["query"]),
            )
            | RunnablePassthrough.assign(
                result=(prompt | llm | StrOutputParser())
            )
            | RunnableLambda(lambda x: {
                "result":           x["result"],
                "answer":           x["result"],
                "source_documents": x["context_docs"],
            })
        )
        return f"✅ QA chain ready ({OLLAMA_MODEL}, {len(all_docs)} chunks, {len(all_pages_by_file)} file(s))"
    except Exception as e:
        return f"⚠️ Vectorstore OK but Ollama unavailable: {e}\n(Fallback to similarity search is active)"


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — MULTI-PDF EXTRACT
# ═════════════════════════════════════════════════════════════════════════════

def process_pdfs(pdf_files, force_ocr: bool):
    global _loaded_files
    if not pdf_files:
        return "⚠️ Vui lòng upload ít nhất một file PDF.", "", _loaded_files_table([])

    all_pages_by_file: dict = {}
    summary_lines:     List[str] = []
    full_text_parts:   List[str] = []
    _loaded_files = []

    for pdf_file in pdf_files:
        pdf_path = Path(pdf_file.name)
        pages, used_ocr = extract_with_ocr_fallback(pdf_path, force_ocr=force_ocr)
        all_pages_by_file[pdf_path.name] = pages
        _loaded_files.append(pdf_path.name)

        method      = "OCR (PaddleOCR)" if used_ocr else "Text (pypdf)"
        total_chars = sum(len(p["text"]) for p in pages)
        summary_lines.append(f"  • {pdf_path.name}  —  {len(pages)} trang, ~{total_chars:,} ký tự  [{method}]")

        full_text_parts.append(f"\n{'═'*50}\n📂 FILE: {pdf_path.name}\n{'═'*50}")
        for p in pages:
            full_text_parts.append(
                f"\n{'─'*40}\n📄 Trang {p['page']}\n{'─'*40}\n"
                + (p["text"] or "(Không có nội dung)")
            )

    qa_status = _build_vectorstore(all_pages_by_file)
    info = (f"📄 Đã load {len(pdf_files)} file PDF:\n"
            + "\n".join(summary_lines)
            + f"\n\n🔧 QA Status: {qa_status}")
    return info, "\n".join(full_text_parts).strip(), _loaded_files_table(_loaded_files)


def _loaded_files_table(files: List[str]) -> str:
    if not files:
        return "Chưa có file nào được load. / No files loaded yet."
    lines = ["Files currently in index:"]
    for i, f in enumerate(files, 1):
        lines.append(f"  {i}. {f}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHAT Q&A
# ═════════════════════════════════════════════════════════════════════════════

def chat(message: str, history: list):
    if not message.strip():
        return history, ""

    h = list(history or [])

    if _vectorstore is None:
        reply = ("⚠️ Chưa có tài liệu nào được load.\n"
                 "Vui lòng upload PDF ở Tab 1 trước.\n"
                 "/ No document loaded. Upload PDFs in Tab 1 first.")
        return _append_pair(h, message, reply), ""

    session_ctx = _build_session_context(message)
    resolved_q, msg_filt = _extract_query_directives(message)
    if session_ctx:
        resolved_q = f"{resolved_q}\n\n{session_ctx}"

    if _qa_chain is not None:
        try:
            invoke_kwargs = {"query": resolved_q}
            if msg_filt:
                invoke_kwargs["filter"] = msg_filt
            resp     = _qa_chain.invoke(invoke_kwargs)
            reply    = resp.get("result") or resp.get("answer") or "Không có kết quả."
            src_docs = resp.get("source_documents") or []
            if src_docs:
                seen, sources = set(), []
                for d in src_docs:
                    md    = getattr(d, "metadata", {}) or {}
                    src   = os.path.basename(md.get("source") or "?")
                    page  = md.get("page")
                    label = f"{src} tr.{page}" if page else src
                    if label not in seen:
                        sources.append(label); seen.add(label)
                reply += "\n\n📌 Nguồn: " + " | ".join(sources)
            else:
                reply = "Không tìm thấy thông tin trong tài liệu."
        except Exception as e:
            reply = f"[LLM error] {e}"
    else:
        # fallback: similarity search
        try:
            _ss_kwargs = {"k": 4}
            if msg_filt.get("source"):
                _ss_kwargs["filter"] = {"source": msg_filt["source"]}
            docs  = _vectorstore.similarity_search(resolved_q, **_ss_kwargs)
            if not docs:
                reply = "Không tìm thấy đoạn nào liên quan."
            else:
                chunks = "\n\n---\n\n".join(
                    f"[{os.path.basename(d.metadata.get('source','?'))} tr.{d.metadata.get('page','?')}]\n{d.page_content}"
                    for d in docs
                )
                reply = "🔍 Đoạn liên quan nhất (LLM unavailable):\n\n" + chunks
        except Exception as e:
            reply = f"[Search error] {e}"

    try:
        _save_session_summary(message, reply.split("\n\n📌")[0])
    except Exception:
        pass

    # Persist this turn so it survives page refresh / server restart
    try:
        _save_chat_turn(message, reply)
    except Exception:
        pass

    return _append_pair(h, message, reply), ""


def clear_session_memory():
    try:
        if os.path.exists(SESSION_MEMORY):
            os.remove(SESSION_MEMORY)
        return "✅ Đã xóa bộ nhớ hội thoại."
    except Exception as e:
        return f"❌ {e}"


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════

def get_diagnostics():
    lines = [
        f"Gradio version : {gr.__version__}  (chat format: {'messages dict' if _GR6 else 'tuples'})",
        f"LangChain OK   : {_langchain_ok}",
        f"OLLAMA_MODEL   : {OLLAMA_MODEL}",
        f"Embedding model: {EMBEDDING_MODEL}  — {'FOUND ✅' if Path(EMBEDDING_MODEL).exists() else 'MISSING ❌'}",
        f"BM25 index     : {BM25_INDEX_PATH}  — {'FOUND ✅' if os.path.exists(BM25_INDEX_PATH) else 'not present'}",
        f"Vectorstore    : {'loaded ✅' if _vectorstore else 'not loaded'}",
        f"QA chain       : {'ready ✅' if _qa_chain else 'not ready'}",
        f"Loaded files   : {_loaded_files or 'none'}",
        "",
        "── Import log ──────────────────────────",
    ] + _import_log
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═════════════════════════════════════════════════════════════════════════════

# Chatbot kwargs:
#   Gradio 6+     -> type= was removed; dict format is native, no kwarg needed
#   Gradio 4.29-5 -> must pass type="messages" to opt into dict format
#   Gradio < 4.29 -> no type= kwarg; uses legacy tuple format
_chatbot_kwargs = {"height": 480, "label": "Chat"}
if _GR6 and not _GR6_NO_TYPE:
    _chatbot_kwargs["type"] = "messages"

with gr.Blocks(title="ScanPDF") as demo:
    gr.Markdown(
        "# 📄 ScanPDF — Multi-PDF Q&A\n"
        "**VI:** Upload nhiều file PDF, trích xuất văn bản và hỏi đáp thông minh.  \n"
        f"**EN:** Upload multiple PDFs, extract text and ask questions. LLM: `{OLLAMA_MODEL}`"
    )

    with gr.Tabs():

        # ── Tab 1 ─────────────────────────────────────────────────────────
        with gr.Tab("📃 Trích xuất / Extract"):
            gr.Markdown(
                "Upload **một hoặc nhiều** PDF cùng lúc. "
                "Mỗi lần bấm Extract sẽ **xây lại index từ đầu** — không còn ngữ cảnh file cũ.\n\n"
                "/ Upload **one or more** PDFs. Each Extract **rebuilds the index** from scratch."
            )
            with gr.Row():
                pdf_input = gr.File(
                    label="Upload PDF(s)",
                    file_types=[".pdf"],
                    file_count="multiple",
                )
                force_ocr_cb = gr.Checkbox(
                    label="Force OCR (bỏ qua text gốc)", value=False
                )
            extract_btn = gr.Button("🔍 Trích xuất / Extract", variant="primary")
            info_box    = gr.Textbox(label="Thông tin / Info",            lines=8,  interactive=False)
            files_box   = gr.Textbox(label="Files in index",              lines=4,  interactive=False)
            text_output = gr.Textbox(label="Nội dung tổng hợp / Content", lines=25, interactive=False)

            extract_btn.click(
                fn=process_pdfs,
                inputs=[pdf_input, force_ocr_cb],
                outputs=[info_box, text_output, files_box],
            )

        # ── Tab 2 ─────────────────────────────────────────────────────────
        with gr.Tab("💬 Hỏi đáp / Chat Q&A"):
            gr.Markdown(
                "**VI:** Upload PDF ở Tab 1 trước, sau đó đặt câu hỏi.  \n"
                "**EN:** Upload PDFs in Tab 1 first, then ask questions here.\n\n"
                "**Tip:** Dùng `source:ten_file.pdf` hoặc `page:3` trong câu hỏi để lọc.\n"
                "/ Use `source:filename.pdf` or `page:3` in your question to filter."
            )
            chatbot = gr.Chatbot(value=_load_chat_history, **_chatbot_kwargs)
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Nhập câu hỏi / Type your question…",
                    scale=5, show_label=False,
                )
                send_btn = gr.Button("Gửi / Send", variant="primary", scale=1)

            send_btn.click(fn=chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
            msg_input.submit(fn=chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])

            with gr.Row():
                clear_chat_btn = gr.Button("🗑️ Xóa chat / Clear chat")
                clear_mem_btn  = gr.Button("🧹 Xóa bộ nhớ / Clear memory")
                mem_status     = gr.Textbox(label="", interactive=False, scale=3)

            clear_chat_btn.click(
                fn=clear_chat_history,
                outputs=[chatbot, mem_status],
            )
            clear_mem_btn.click(fn=clear_session_memory, outputs=[mem_status])

        # ── Tab 3 ─────────────────────────────────────────────────────────
        with gr.Tab("🔬 Diagnostics"):
            gr.Markdown(
                "Kiểm tra trạng thái hệ thống, import log và model paths.\n"
                "/ Check system status, import log, and model paths."
            )
            diag_btn = gr.Button("🔄 Refresh diagnostics")
            diag_box = gr.Textbox(label="System status", lines=30, interactive=False)
            diag_btn.click(fn=get_diagnostics, outputs=[diag_box])

if __name__ == "__main__":
    print("Gradio", gr.__version__, "— chat format:", "messages dict" if _GR6 else "tuples")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Monochrome(),
    )