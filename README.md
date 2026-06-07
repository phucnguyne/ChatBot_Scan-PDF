# QA Bot — PDF Question Answering (Offline)

An offline question-answering assistant that answers Vietnamese questions from PDF documents using local models (embeddings + LLM).

This repository helps you:
- Download and prepare an embedding model and a local LLM (GGUF format).
- Convert PDFs into vector embeddings and store them in a FAISS vector store.
- Run a simple REPL to ask questions against the PDF knowledge base.

---

## Project structure

```
.
├── data/                        # Put your PDF files here
├── models/                      # Models saved in GGUF format (downloaded by helpers)
├── Embedding/
│   └── prepare_vector_db.py     # Create FAISS DB from PDFs
├── prepare_model/
│   ├── model_embedding.py       # Download / prepare embedding model
│   └── model_llm.py             # Download / prepare LLM model
├── vectorstores/
│   └── db_faiss/                # Generated FAISS vector database
├── qabot.py                     # Main QA program (REPL)
├── setup.txt                    # Python dependencies (pip install -r setup.txt)
└── README.md                    # This file
```

---

## Requirements

- Python 3.9+
- 8 GB RAM minimum (16 GB recommended)
- ~6 GB free disk space (LLM model ~5GB)
- Internet only required to download models the first time

If you have an NVIDIA GPU and want to use CUDA, install `ctransformers[cuda]` instead of `ctransformers`.

---

## Quick start

1. Create a virtual environment and activate it (recommended):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r setup.txt
```

3. Download the embedding model:

```bash
python prepare_model/model_embedding.py
```

4. Download the LLM model:

```bash
python prepare_model/model_llm.py
```

5. Add your PDF files to `data/`.

6. Build the vector database:

```bash
python Embedding/prepare_vector_db.py
```

7. Start the QA REPL:

```bash
python qabot.py
```

Type questions in Vietnamese; enter `exit` to quit.

8. Run the test suite:

```bash
python -m unittest discover -s tests
```

---

## How it works (overview)

- Prepare: `prepare_model/*` downloads the embedding model and the local LLM in GGUF format into `models/`.
- Indexing: `Embedding/prepare_vector_db.py` reads PDFs from `data/`, splits text into chunks, computes embeddings, and stores them in `vectorstores/db_faiss/`.
- Query: `qabot.py` takes a user question, finds relevant chunks in FAISS, and passes context + question to the local LLM to generate an answer.

---

## Session memory format

`vectorstores/session_memory.jsonl` is append-only JSONL. Each line is a JSON object with these fields:

- `summary`: short one-line recap of the turn.
- `topic`: best-effort topic extracted from the question.
- `question`: original user question.
- `answer`: truncated answer text stored for follow-up context.
- `created_at`: UTC timestamp in ISO-8601 format.

`qabot.py` reads the most recent entries from this file to resolve follow-up questions like “Môn đó” or “câu đó”.

---

## Troubleshooting

- FAISS load error about deserialization: change the load call to allow dangerous deserialization (example):

```python
db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
```

- Missing model files: ensure `models/` contains the required `.gguf` files. Re-run `prepare_model/*.py` if needed.

- Empty or missing Vector DB: run `Embedding/prepare_vector_db.py` and confirm `data/` contains PDFs.

---

## Notes & Next steps

- This repo is designed to run fully offline after models are downloaded.
- You can extend `qabot.py` to add logging, streaming responses, or a web UI.

---

## Vietnamese (Tóm tắt nhanh)

Hệ thống hỏi đáp offline bằng tiếng Việt từ file PDF. Làm theo các bước: cài thư viện (`pip install -r setup.txt`), tải mô hình (`prepare_model/*.py`), thêm PDF vào `data/`, tạo vector DB (`Embedding/prepare_vector_db.py`), rồi chạy `python qabot.py`.

---

If you want, I can also:
- Add a short CONTRIBUTING guide
- Create a `requirements.txt` or `pyproject.toml`
- Make `qabot.py` print clearer startup instructions

