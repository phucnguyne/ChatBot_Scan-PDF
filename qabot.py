"""CLI question-answering bot for the local PDF knowledge base."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    CHAT_HISTORY_PATH,
    BM25_INDEX_PATH,
    OLLAMA_MODEL,
    TOP_K_RERANK,
    TOP_K_LONG_ANSWER,
    TOP_K_RETRIEVAL,
    VECTOR_DB_PATH,
)
from memory.history_manager import clear_history, load_history, save_turn
from rag.citation import build_citation_string
from retrieval.metadata_filter import extract_directives

LONG_ANSWER_TERMS = re.compile(r"\b(?:all|complete|entire|detailed|explain|list)\b", re.IGNORECASE)


@dataclass
class SearchResult:
    content: str
    metadata: dict[str, Any]
    score: float = 0.0


def format_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks with source metadata for the LLM."""
    sections = []
    for result in results:
        source = os.path.basename(str(result.metadata.get("source", "unknown")))
        page = result.metadata.get("page")
        location = f"{source} - page {page}" if page is not None else source
        sections.append(f"[Source: {location}]\n{result.content.strip()}")
    return "\n\n".join(sections)


def build_prompt(question: str, context: str, history: list[dict[str, str]] | None = None) -> str:
    """Build a grounded prompt with a small amount of chat context."""
    long_answer = bool(LONG_ANSWER_TERMS.search(question))
    recent_history = [] if long_answer else (history[-4:] if history else [])
    history_text = "\n".join(
        f"{item['role']}: {item['content']}" for item in recent_history
    )
    return f"""You are a PDF question-answering assistant.
Answer only using the CONTEXT. If the CONTEXT does not contain the answer, say:
\"The document does not contain this information.\"
Do not make up facts or guess. Cite sources using the format
[Source: filename.pdf - page N] when possible. For requests asking for a complete
answer, cover every relevant subsection in the CONTEXT; do not stop at the first one.
Answer in the same language as the user's question.

RECENT CHAT HISTORY:
{history_text or '(none)'}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def append_missing_sections(answer: str, question: str, results: list[SearchResult]) -> str:
    """Prevent a small model from silently omitting numbered sub-sections."""
    match = re.search(r"\b(?:question|section)\s+(\d+)\b", question, re.IGNORECASE)
    if not match or not LONG_ANSWER_TERMS.search(question):
        return answer
    section = match.group(1)
    source_text = "\n".join(result.content for result in results)
    headings = sorted(set(re.findall(rf"(?m)^\s*({re.escape(section)}\.\d+)\.", source_text)))
    missing = [heading for heading in headings if not re.search(rf"(?m)^\s*{re.escape(heading)}\.", answer)]
    if not missing:
        return answer
    additions = []
    for heading in missing:
        pattern = rf"(?ms)^\s*{re.escape(heading)}\..*?(?=^\s*\d+(?:\.\d+)*\.|\Z)"
        excerpt = re.search(pattern, source_text)
        if excerpt:
            additions.append(excerpt.group(0).strip())
    if additions:
        return answer.rstrip() + "\n\n" + "\n\n".join(additions)
    return answer


class QABot:
    """Hybrid FAISS + BM25 PDF assistant backed by Ollama."""

    def __init__(self, vector_db_path: str = VECTOR_DB_PATH, bm25_index_path: str = BM25_INDEX_PATH,
                 history_path: str = CHAT_HISTORY_PATH) -> None:
        self.vector_db_path = vector_db_path
        self.bm25_index_path = bm25_index_path
        self.history_path = history_path
        self._faiss = None
        self._embedding = None
        self._llm = None

    @staticmethod
    def _matches_filter(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        if not filters:
            return True
        source = os.path.basename(str(metadata.get("source", "")))
        if "source" in filters and source.lower() != str(filters["source"]).lower():
            return False
        return "page" not in filters or metadata.get("page") == filters["page"]

    def _load_faiss(self) -> None:
        if self._faiss is not None:
            return
        if not Path(self.vector_db_path).exists():
            raise FileNotFoundError(
                f"No vector database found at {VECTOR_DB_PATH}. "
                "Import a PDF into documents/ and build the index in the web UI."
            )
        from ingestion.index_builder import SentenceTransformerEmbedding
        from langchain_community.vectorstores import FAISS

        self._embedding = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
        self._faiss = FAISS.load_local(
            self.vector_db_path,
            self._embedding,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, question: str, filters: dict[str, Any]) -> list[SearchResult]:
        self._load_faiss()
        long_answer = bool(LONG_ANSWER_TERMS.search(question))
        retrieval_k = TOP_K_LONG_ANSWER if long_answer else TOP_K_RETRIEVAL
        semantic_docs = self._faiss.similarity_search(question, k=retrieval_k)
        results: list[SearchResult] = []
        seen: set[str] = set()

        for document in semantic_docs:
            metadata = dict(document.metadata or {})
            if self._matches_filter(metadata, filters):
                key = document.page_content.strip()
                if key not in seen:
                    results.append(SearchResult(document.page_content, metadata))
                    seen.add(key)

        try:
            from retrieval.bm25_search import bm25_search

            for item in bm25_search(question, retrieval_k, self.bm25_index_path):
                metadata = dict(item.get("metadata") or {})
                content = str(item.get("content", ""))
                if self._matches_filter(metadata, filters) and content.strip() not in seen:
                    results.append(SearchResult(content, metadata, float(item.get("score", 0))))
                    seen.add(content.strip())
        except FileNotFoundError:
            pass

        if long_answer and results:
            # A section often spans adjacent chunks. Add neighbors so a request
            # such as "explain all of question 6" includes 6.1, 6.2, etc.
            by_index = {int(r.metadata.get("chunk_index", -1)): r for r in results}
            expanded = dict(by_index)
            for index in list(by_index):
                for neighbor in range(index - 2, index + 3):
                    if neighbor not in expanded:
                        for candidate in semantic_docs:
                            metadata = dict(candidate.metadata or {})
                            if metadata.get("chunk_index") == neighbor:
                                expanded[neighbor] = SearchResult(candidate.page_content, metadata)
                                break
            results = [expanded[index] for index in sorted(expanded)]
            return results[:TOP_K_LONG_ANSWER]
        return results[:TOP_K_RERANK]

    def _load_llm(self) -> None:
        if self._llm is None:
            from langchain_ollama import ChatOllama

            self._llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_predict=2048)

    def answer(self, question: str) -> tuple[str, str]:
        cleaned_question, filters = extract_directives(question)
        if not cleaned_question:
            return "Please enter a specific question.", ""
        results = self.retrieve(cleaned_question, filters)
        if not results:
            return "The document does not contain this information.", ""

        self._load_llm()
        prompt = build_prompt(cleaned_question, format_context(results), load_history(self.history_path))
        response = self._llm.invoke(prompt)
        answer = str(getattr(response, "content", response)).strip()
        answer = answer.replace("Answer based on CONTEXT:", "").strip()
        answer = append_missing_sections(answer, cleaned_question, results)
        citation = build_citation_string(results)
        save_turn(question, answer, self.history_path)
        return answer, citation


def print_help() -> None:
    print("Commands: /help, /clear (clear history), /sources (show index), /reload, exit")


def run_repl() -> None:
    print(f"QA Bot ready | Ollama model: {OLLAMA_MODEL}")
    print("Enter a question. Type /help to see commands or exit to quit.")
    bot = QABot()

    while True:
        try:
            question = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.lower() in {"exit", "quit", ":q"}:
            print("Goodbye!")
            break
        if question == "/help":
            print_help()
            continue
        if question == "/clear":
            clear_history()
            print("Chat history cleared.")
            continue
        if question == "/sources":
            print(f"Vector DB: {VECTOR_DB_PATH}\nHistory: {CHAT_HISTORY_PATH}")
            continue
        if question == "/reload":
            bot = QABot()
            print("The model and index will be reloaded on the next question.")
            continue
        if not question:
            continue

        try:
            answer, citation = bot.answer(question)
            print(f"\nBot > {answer}")
            if citation:
                print(citation)
        except Exception as error:
            print(f"\nError: {error}")
            print("Check setup.txt, Ollama (ollama serve), and the PDF index.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about PDF documents")
    parser.add_argument("--question", "-q", help="Ask one question and exit")
    args = parser.parse_args()
    if args.question:
        try:
            answer, citation = QABot().answer(args.question)
            print(answer)
            if citation:
                print(citation)
        except Exception as error:
            parser.error(str(error))
        return
    run_repl()


if __name__ == "__main__":
    main()