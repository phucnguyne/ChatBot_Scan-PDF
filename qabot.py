import os
import pickle
import json
from typing import List, Optional

from langchain_community.llms import CTransformers
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

# Cau hinh
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"
bm25_index_path = os.path.join(vector_db_path, "bm25_index.pkl")
SESSION_MEMORY = os.path.join("vectorstores", "session_memory.jsonl")

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

# Tao prompt template
def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt


# Tao simple chain
def create_qa_chain(prompt, llm, db):
    # Default simple retriever (will be replaced with a custom one if needed)
    retriever = db.as_retriever(search_kwargs={"k":3}, max_tokens_limit=1024)
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    return llm_chain


class CustomRetriever:
    """Custom retriever that supports metadata filtering, optional BM25 hybrid search and reranking."""

    def __init__(self, db: FAISS, bm25_path: Optional[str] = None, reranker_model: Optional[str] = None, fetch_k: int = 20, top_k: int = 5):
        self.db = db
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.bm25 = None
        self.bm25_metadatas = None
        if bm25_path and os.path.exists(bm25_path):
            try:
                with open(bm25_path, "rb") as fh:
                    data = pickle.load(fh)
                    self.bm25 = data.get("bm25")
                    self.bm25_metadatas = data.get("metadatas")
                    self.bm25_docs = data.get("docs")
            except Exception:
                self.bm25 = None

        self.reranker = None
        if reranker_model and CrossEncoder is not None:
            try:
                self.reranker = CrossEncoder(reranker_model)
            except Exception:
                self.reranker = None

    def _filter_by_metadata(self, docs: List, filter: dict):
        if not filter:
            return docs
        out = []
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            ok = True
            for k, v in filter.items():
                if md.get(k) != v:
                    ok = False
                    break
            if ok:
                out.append(d)
        return out

    def get_relevant_documents(self, query: str, **kwargs):
        # accept optional filter in kwargs
        metadata_filter = kwargs.get("filter") or kwargs.get("metadata_filter")

        # 1) FAISS search
        try:
            faiss_docs = self.db.similarity_search(query, k=self.fetch_k)
        except Exception:
            faiss_docs = []

        # 2) optional BM25 candidates
        bm25_candidates = []
        if self.bm25 is not None and self.bm25_docs is not None:
            tokenized = query.split()
            try:
                scores = self.bm25.get_scores(tokenized)
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.fetch_k]
                for idx in top_idx:
                    bm25_candidates.append({"text": self.bm25_docs[idx], "metadata": self.bm25_metadatas[idx]})
            except Exception:
                bm25_candidates = []

        # Merge FAISS results and BM25 (keep unique by chunk_id or text)
        merged = []
        seen = set()
        for d in faiss_docs:
            key = (d.metadata.get("chunk_id"), d.page_content) if hasattr(d, "metadata") else d.page_content
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)
        for bc in bm25_candidates:
            key = (bc.get("metadata", {}).get("chunk_id"), bc.get("text"))
            if key in seen:
                continue
            # create simple Document-like object with page_content and metadata
            class _D:
                pass

            d = _D()
            d.page_content = bc.get("text")
            d.metadata = bc.get("metadata", {})
            merged.append(d)

        # apply metadata filter
        merged = self._filter_by_metadata(merged, metadata_filter)

        # rerank if reranker available
        if self.reranker is not None and merged:
            pairs = [(query, d.page_content) for d in merged]
            try:
                scores = self.reranker.predict(pairs)
                scored = list(zip(merged, scores))
                scored.sort(key=lambda x: x[1], reverse=True)
                merged = [s[0] for s in scored]
            except Exception:
                pass

        return merged[: self.top_k]

# Read tu VectorDB
def read_vectors_db():
    # Embeding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(
        vector_db_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return db


# Bat dau thu nghiem
db = read_vectors_db()
llm = load_llm(model_file)

# Tao Prompt with stricter hallucination guard and citation instruction
template = """<|im_start|>system
Bạn chỉ được phép trả lời dựa trên phần `context` được cung cấp bên dưới. Nếu thông tin không có trong context, hãy trả lời: "Không tìm thấy thông tin trong tài liệu". Không được suy đoán. Khi trả lời, nếu có thể, trích dẫn nguồn theo định dạng: [Nguồn: <tên file> - trang <page>].

{context}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant"""
prompt = creat_prompt(template)

# build custom retriever with BM25 and reranker if available
reranker_model = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
retriever = CustomRetriever(db, bm25_path=bm25_index_path if os.path.exists(bm25_index_path) else None,
                            reranker_model=reranker_model, fetch_k=20, top_k=5)

llm_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)


def _save_session_summary(question: str, answer: str):
    # Lightweight heuristic summary: try to detect noun after 'Môn' else store short Q/A
    summary = None
    q = question.strip()
    if q.lower().startswith("môn "):
        # take until punctuation
        subj = q[4:].split(" ")[:5]
        subj = " ".join(subj)
        summary = f"{subj} -> {answer}"[:200]
    else:
        summary = f"Q: {q} | A: {answer}"[:200]

    os.makedirs(os.path.dirname(SESSION_MEMORY), exist_ok=True)
    with open(SESSION_MEMORY, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"summary": summary}) + "\n")


# Chay cai chain
while True:
    question = input("\nNhập câu hỏi (gõ 'exit' để thoát): ")
    if question.lower() == "exit":
        print("Thoát chương trình.")
        break
    # allow simple metadata filter syntax: e.g. "source:database.pdf"
    meta_filter = None
    if "source:" in question or "page:" in question:
        parts = [p.strip() for p in question.split() if ":" in p]
        for p in parts:
            k, v = p.split(":", 1)
            if k and v:
                if meta_filter is None:
                    meta_filter = {}
                meta_filter[k] = v

    invoke_kwargs = {"query": question}
    if meta_filter:
        invoke_kwargs["filter"] = meta_filter

    response = llm_chain.invoke(invoke_kwargs)
    answer = response.get("result") or response.get("answer") or ""
    print("\nTrả lời:", answer)

    # print citation sources if available
    src_docs = response.get("source_documents") or []
    seen_src = set()
    if src_docs:
        print("\nNguồn:")
        for d in src_docs:
            md = getattr(d, "metadata", {}) or {}
            src = md.get("source") or md.get("file") or md.get("path") or "unknown"
            page = md.get("page")
            label = f"{src}"
            if page:
                label += f" - trang {page}"
            if label not in seen_src:
                print(f"- {label}")
                seen_src.add(label)

    # store session memory summary (one-line)
    try:
        _save_session_summary(question, answer)
    except Exception:
        pass