"""
app.py — Gradio UI for ScanPDF
  Tab 1: Upload PDF → Extract text (pypdf + PaddleOCR fallback)
  Tab 2: Chat Q&A over extracted text (LangChain + FAISS)
"""

import sys
import tempfile
from pathlib import Path

import gradio as gr

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

POPPLER_PATH = r"C:\poppler-26.02.0\Library\bin"  # adjust if needed

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

try:
    from paddleocr import PaddleOCR
    _ocr_engine = None  # lazy init
except Exception:
    PaddleOCR = None
    _ocr_engine = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import GPT4AllEmbeddings
    from langchain_classic.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate
    from langchain_community.llms import CTransformers
    _langchain_ok = True
except Exception as _langchain_err:
    _langchain_ok = False
    print(f"[LangChain import error] {_langchain_err}")

# ── Paths ─────────────────────────────────────────────────────────────────────
VECTOR_DB_PATH  = str(PROJECT_ROOT / "vectorstores" / "db_faiss")
EMBEDDING_MODEL = str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2-f16.gguf")
LLM_MODEL       = str(PROJECT_ROOT / "models" / "vinallama-7b-chat_q5_0.gguf")

PROMPT_TEMPLATE = """<|im_start|>system
Bạn chỉ được phép trả lời dựa trên phần `context` được cung cấp bên dưới. Nếu thông tin không có trong context, hãy trả lời: "Không tìm thấy thông tin trong tài liệu". Không được suy đoán.

{context}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant"""

# ── Global state ──────────────────────────────────────────────────────────────
_vectorstore = None
_qa_chain    = None


# ═════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None and PaddleOCR is not None:
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="vi")
    return _ocr_engine


def extract_text_pages(pdf_path: Path) -> list[dict]:
    """Extract text using pypdf."""
    if PdfReader is None:
        return []
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append({"page": i, "text": text})
    return pages


def ocr_pdf_pages(pdf_path: Path) -> list[dict]:
    """Fallback: render via Poppler → PaddleOCR."""
    if convert_from_path is None or PaddleOCR is None:
        return []
    try:
        images = convert_from_path(str(pdf_path), dpi=200, poppler_path=POPPLER_PATH)
    except Exception as e:
        print(f"[Poppler] {e}")
        return []

    ocr = _get_ocr()
    if ocr is None:
        return []

    pages = []
    for i, img in enumerate(images, start=1):
        try:
            result = ocr.ocr(img, cls=True)
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
        pages.append({"page": i, "text": "\n".join(lines).strip()})
    return pages


def extract_with_ocr_fallback(pdf_path: Path, force_ocr: bool = False):
    pages = extract_text_pages(pdf_path)
    has_text = any(p["text"] for p in pages)
    if has_text and not force_ocr:
        return pages, False
    ocr_pages = ocr_pdf_pages(pdf_path)
    if ocr_pages:
        return ocr_pages, True
    return pages, False


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXTRACT TEXT
# ═════════════════════════════════════════════════════════════════════════════

def process_pdf(pdf_file, force_ocr: bool):
    """Called by Gradio when user uploads a PDF."""
    if pdf_file is None:
        return "⚠️ Vui lòng upload file PDF. / Please upload a PDF file.", ""

    pdf_path = Path(pdf_file.name)
    pages, used_ocr = extract_with_ocr_fallback(pdf_path, force_ocr=force_ocr)

    method = "OCR (PaddleOCR)" if used_ocr else "Text extraction (pypdf)"
    info = (
        f"📄 File: {pdf_path.name}\n"
        f"📃 Pages: {len(pages)}\n"
        f"🔍 Method: {method}"
    )

    full_text = ""
    for p in pages:
        full_text += f"\n{'─'*40}\n📄 Trang / Page {p['page']}\n{'─'*40}\n"
        full_text += p["text"] or "(Không có nội dung / No content)"

    # Build vectorstore for Tab 2
    _build_vectorstore(pages, pdf_path.name)

    return info, full_text.strip()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHAT Q&A
# ═════════════════════════════════════════════════════════════════════════════

def _build_vectorstore(pages: list[dict], source_name: str):
    """
    Ưu tiên load FAISS đã build sẵn từ vectorstores/db_faiss.
    Nếu chưa có, build mới từ pages vừa extract.
    Dùng GPT4AllEmbeddings + CTransformers (đúng với qabot.py).
    """
    global _vectorstore, _qa_chain
    if not _langchain_ok:
        print("[QA] LangChain không khả dụng")
        return

    import os
    from pathlib import Path as _Path

    embedding_model_file = EMBEDDING_MODEL
    if not _Path(embedding_model_file).exists():
        print(f"[QA] Không tìm thấy embedding model: {embedding_model_file}")
        return

    try:
        embeddings = GPT4AllEmbeddings(model_file=embedding_model_file)
    except Exception as e:
        print(f"[QA] GPT4AllEmbeddings lỗi: {e}")
        return

    # 1) Thử load FAISS đã build sẵn
    if os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
        try:
            _vectorstore = FAISS.load_local(
                VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True
            )
            print(f"[QA] Loaded FAISS từ {VECTOR_DB_PATH}")
        except Exception as e:
            print(f"[QA] Load FAISS sẵn thất bại: {e}")
            _vectorstore = None

    # 2) Nếu không có, build mới từ pages
    if _vectorstore is None:
        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        if not full_text.strip():
            print("[QA] Không có text để build vectorstore")
            return
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
            docs = splitter.create_documents(
                [full_text],
                metadatas=[{"source": source_name}]
            )
            _vectorstore = FAISS.from_documents(docs, embeddings)
            print("[QA] Build FAISS mới từ PDF vừa upload")
        except Exception as e:
            print(f"[QA] Build FAISS thất bại: {e}")
            _vectorstore = None
            return

    # 3) Load LLM CTransformers (vinallama)
    _qa_chain = None
    llm_path = _Path(LLM_MODEL)
    if not llm_path.exists():
        print(f"[QA] Không tìm thấy LLM model: {LLM_MODEL} — chỉ dùng similarity search")
        return

    try:
        llm = CTransformers(
            model=str(llm_path),
            model_type="llama",
            config={
                "max_new_tokens": 256,   # giảm từ 1024 → 256 (nhanh ~4x)
                "temperature": 0.3,      # tăng nhẹ để tránh greedy search chậm
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.1,
                "context_length": 2048,  # giới hạn context window
                "batch_size": 1,
                "threads": 4,            # dùng 4 CPU threads
            },
        )
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        _qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        print("[QA] LLM chain sẵn sàng")
    except Exception as e:
        print(f"[QA] Load LLM thất bại: {e} — chỉ dùng similarity search")
        _qa_chain = None


def chat(message: str, history: list):
    if not message.strip():
        return history, ""

    if _vectorstore is None:
        reply = (
            "⚠️ Chưa có tài liệu nào được load. Vui lòng upload PDF ở Tab 1 trước.\n"
            "/ No document loaded yet. Please upload a PDF in Tab 1 first."
        )
    elif _qa_chain is not None:
        try:
            response = _qa_chain.invoke({"query": message})
            reply = response.get("result") or response.get("answer") or ""
            # Thêm nguồn trích dẫn
            src_docs = response.get("source_documents") or []
            if src_docs:
                seen = set()
                sources = []
                for d in src_docs:
                    md = getattr(d, "metadata", {}) or {}
                    src = md.get("source") or "unknown"
                    page = md.get("page")
                    label = f"{src} - trang {page}" if page else src
                    if label not in seen:
                        sources.append(label)
                        seen.add(label)
                reply += "\n\n📌 Nguồn: " + ", ".join(sources)
        except Exception as e:
            reply = f"[LLM error] {e}"
    else:
        try:
            docs = _vectorstore.similarity_search(message, k=3)
            chunks = "\n\n---\n\n".join(d.page_content for d in docs)
            reply = "🔍 Các đoạn liên quan nhất / Most relevant passages:\n\n" + chunks
        except Exception as e:
            reply = f"[Search error] {e}"

    history = list(history or [])
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, ""


# ═════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="ScanPDF") as demo:
    gr.Markdown(
        """
        # 📄 ScanPDF
        **VI:** Upload PDF để trích xuất văn bản và hỏi đáp thông minh.  
        **EN:** Upload a PDF to extract text and ask questions about it.
        """
    )

    with gr.Tabs():
        # ── Tab 1: Extract ────────────────────────────────────────────────
        with gr.Tab("📃 Trích xuất / Extract"):
            with gr.Row():
                pdf_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                )
                force_ocr_checkbox = gr.Checkbox(
                    label="Force OCR (bỏ qua text gốc / ignore embedded text)",
                    value=False,
                )
            extract_btn = gr.Button("🔍 Trích xuất / Extract", variant="primary")

            info_box = gr.Textbox(
                label="Thông tin / Info",
                lines=3,
                interactive=False,
            )
            text_output = gr.Textbox(
                label="Nội dung / Content",
                lines=20,
                interactive=False,
            )

            extract_btn.click(
                fn=process_pdf,
                inputs=[pdf_input, force_ocr_checkbox],
                outputs=[info_box, text_output],
            )

        # ── Tab 2: Chat ───────────────────────────────────────────────────
        with gr.Tab("💬 Hỏi đáp / Chat Q&A"):
            gr.Markdown(
                "**VI:** Upload PDF ở Tab 1 trước, sau đó đặt câu hỏi tại đây.  \n"
                "**EN:** Upload a PDF in Tab 1 first, then ask questions here."
            )
            chatbot = gr.Chatbot(height=400, label="Chat")
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Nhập câu hỏi / Type your question...",
                    scale=5,
                    show_label=False,
                )
                send_btn = gr.Button("Gửi / Send", variant="primary", scale=1)

            send_btn.click(
                fn=chat,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input],
            )
            msg_input.submit(
                fn=chat,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input],
            )

            clear_btn = gr.Button("🗑️ Xóa / Clear")
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )