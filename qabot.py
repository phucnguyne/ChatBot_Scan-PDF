import os
import pickle
import json
import re
import math
from datetime import datetime
from typing import List, Optional

from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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


def _normalize_metadata_filter_key(key: str) -> str:
    key = key.strip().lower()
    if key in {"file", "path"}:
        return "source"
    return key


def _normalize_metadata_filter_value(key: str, value: str):
    value = value.strip().strip("\"'")
    if key == "source":
        return os.path.basename(value)
    if key == "page" and value.isdigit():
        return int(value)
    return value


def _extract_query_directives(question: str):
    cleaned_parts = []
    metadata_filter = {}
    for token in question.split():
        if ":" not in token:
            cleaned_parts.append(token)
            continue

        key, value = token.split(":", 1)
        norm_key = _normalize_metadata_filter_key(key)
        if norm_key in {"source", "page"} and value:
            metadata_filter[norm_key] = _normalize_metadata_filter_value(norm_key, value)
            continue

        cleaned_parts.append(token)

    cleaned_question = " ".join(cleaned_parts).strip()
    return cleaned_question, metadata_filter


def _metadata_matches_filter(metadata: dict, metadata_filter: dict) -> bool:
    if not metadata_filter:
        return True

    md = metadata or {}
    for key, expected_value in metadata_filter.items():
        actual_value = md.get(key)
        if key == "source" and isinstance(actual_value, str):
            actual_value = os.path.basename(actual_value)
        if key == "page" and isinstance(actual_value, str) and actual_value.isdigit():
            actual_value = int(actual_value)
        if actual_value != expected_value:
            return False
    return True


def _bm25_score_query(query_tokens: List[str], bm25_index: dict) -> List[float]:
    doc_token_freqs = bm25_index.get("doc_token_freqs") or []
    doc_lengths = bm25_index.get("doc_lengths") or []
    avgdl = bm25_index.get("avgdl") or 0.0
    idf = bm25_index.get("idf") or {}
    k1 = bm25_index.get("k1", 1.5)
    b = bm25_index.get("b", 0.75)

    scores = [0.0 for _ in doc_token_freqs]
    if not query_tokens or not doc_token_freqs:
        return scores

    for token in query_tokens:
        token_idf = idf.get(token)
        if token_idf is None:
            continue
        for index, token_freqs in enumerate(doc_token_freqs):
            frequency = token_freqs.get(token, 0)
            if not frequency:
                continue
            doc_length = doc_lengths[index] if index < len(doc_lengths) else 0
            denominator = frequency + k1 * (1 - b + b * (doc_length / avgdl)) if avgdl else frequency + k1
            scores[index] += token_idf * (frequency * (k1 + 1)) / denominator

    return scores


def _extract_topic(question: str) -> str:
    text = question.strip()
    if not text:
        return ""

    lowered = text.lower()
    for prefix in ("môn ", "bài ", "câu ", "chủ đề ", "phần "):
        if lowered.startswith(prefix):
            remainder = text[len(prefix):].strip()
            if not remainder:
                return ""
            words = remainder.split()
            if not words:
                return ""
            if words[0].lower() in {"đó", "này", "ấy", "kia", "nữa"}:
                return ""
            return " ".join(words[:5]).strip(" .,!?:;\"'")

    match = re.match(r"^([A-Za-zÀ-ỹ0-9_-]{2,})(?:\s+(.+))?$", text)
    if match:
        head = match.group(1).strip()
        tail = (match.group(2) or "").split()
        if tail and tail[0].lower() not in {"đó", "này", "ấy", "kia", "nữa"}:
            return " ".join([head] + tail[:4]).strip(" .,!?:;\"'")

    return ""


def _load_session_entries(limit: int = 20) -> List[dict]:
    if not os.path.exists(SESSION_MEMORY):
        return []

    entries: List[dict] = []
    try:
        with open(SESSION_MEMORY, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict) and item.get("summary"):
                    entries.append(item)
    except Exception:
        return []

    return entries[-limit:]


def _load_session_context(n: int = 5) -> str:
    entries = _load_session_entries(n)
    if not entries:
        return ""

    lines = []
    for entry in entries:
        summary = (entry.get("summary") or "").strip()
        if summary:
            lines.append(summary)

    return "\n".join(lines)


def _needs_session_context(question: str) -> bool:
    q = question.lower()
    return any(
        marker in q
        for marker in (
            "môn đó",
            "bài đó",
            "câu đó",
            "cái đó",
            "nó",
            "đó",
            "tiếp theo",
            "lúc nãy",
            "vừa rồi",
            "câu trước",
        )
    )


def _build_session_context(question: str) -> str:
    entries = _load_session_entries()
    if not entries:
        return ""

    if _needs_session_context(question):
        relevant_entries = entries[-1:]
    else:
        relevant_entries = entries[-3:]

    latest_entry = entries[-1]
    summaries = [entry.get("summary", "").strip() for entry in relevant_entries if entry.get("summary")]
    if not summaries:
        return ""

    context_lines = []
    topic = (latest_entry.get("topic") or "").strip()
    if topic:
        context_lines.append(f"Chủ đề gần nhất: {topic}")

    last_question = (latest_entry.get("question") or "").strip()
    if last_question:
        context_lines.append(f"Câu hỏi gần nhất: {last_question}")

    context_lines.extend(f"- {summary}" for summary in summaries)
    return "Ngữ cảnh hội thoại trước đó:\n" + "\n".join(context_lines)

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
                    self.bm25 = data
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
        return [d for d in docs if _metadata_matches_filter(getattr(d, "metadata", {}) or {}, filter)]

    def get_relevant_documents(self, query: str, **kwargs):
        # accept optional filter in kwargs
        cleaned_query, parsed_filter = _extract_query_directives(query)
        metadata_filter = kwargs.get("filter") or kwargs.get("metadata_filter") or parsed_filter
        search_query = cleaned_query or query

        # 1) FAISS search
        try:
            faiss_docs = self.db.similarity_search(search_query, k=self.fetch_k)
        except Exception:
            faiss_docs = []

        # 2) optional BM25 candidates
        bm25_candidates = []
        if self.bm25 is not None and self.bm25_docs is not None:
            tokenized = search_query.split()
            try:
                if hasattr(self.bm25, "get_scores"):
                    scores = self.bm25.get_scores(tokenized)
                else:
                    scores = _bm25_score_query(tokenized, self.bm25)
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
            pairs = [(search_query, d.page_content) for d in merged]
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
    q = question.strip()
    topic = _extract_topic(q)
    if q.lower().startswith("môn "):
        # take until punctuation
        subj = q[4:].split(" ")[:5]
        subj = " ".join(subj)
        summary = f"{subj} -> {answer}"[:200]
    else:
        summary = f"Q: {q} | A: {answer}"[:200]

    os.makedirs(os.path.dirname(SESSION_MEMORY), exist_ok=True)
    with open(SESSION_MEMORY, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "summary": summary,
                    "topic": topic,
                    "question": q,
                    "answer": answer[:500],
                    "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                },
                ensure_ascii=False,
            )
            + "\n"
        )


# Chay cai chain
while True:
    question = input("\nNhập câu hỏi (gõ 'exit' để thoát): ")
    if question.lower() == "exit":
        print("Thoát chương trình.")
        break
    session_context = _build_session_context(question)
    resolved_question, _ = _extract_query_directives(question)
    if session_context:
        resolved_question = f"{resolved_question}\n\n{session_context}"

    invoke_kwargs = {"query": resolved_question}

    response = llm_chain.invoke(invoke_kwargs)
    answer = response.get("result") or response.get("answer") or ""

    # print citation sources if available
    src_docs = response.get("source_documents") or []
    if not src_docs:
        answer = "Không tìm thấy thông tin trong tài liệu"

    print("\nTrả lời:", answer)

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