"""Local web UI server with isolated files for every conversation."""

from __future__ import annotations

import cgi
import json
import shutil
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from build_index import build_index
from config import OLLAMA_MODEL
from memory.history_manager import clear_history, load_history
from qabot import QABot

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "ui"
CHAT_ROOT = PROJECT_ROOT / "workspaces"
REGISTRY_PATH = CHAT_ROOT / "chats.json"
CHAT_ROOT.mkdir(exist_ok=True)


def paths_for(chat_id: str) -> dict[str, Path]:
    root = CHAT_ROOT / chat_id
    index = root / "vectorstores" / "db_faiss"
    return {"root": root, "documents": root / "documents", "vector": index,
            "bm25": index / "bm25_index.pkl", "manifest": index / "index_manifest.json",
            "history": root / "history.jsonl"}


def registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_registry(data: dict) -> None:
    CHAT_ROOT.mkdir(exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create_chat(title: str = "Cuộc trò chuyện mới") -> str:
    chat_id = uuid.uuid4().hex[:12]
    paths = paths_for(chat_id)
    paths["documents"].mkdir(parents=True, exist_ok=True)
    data = registry()
    data[chat_id] = {"title": title, "created": time.time()}
    save_registry(data)
    return chat_id


def matching_index(paths: dict[str, Path]) -> bool:
    documents = sorted(paths["documents"].glob("*.pdf")) if paths["documents"].exists() else []
    if not documents or not paths["manifest"].exists():
        return False
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        current = [{"name": p.name, "size": p.stat().st_size, "modified": p.stat().st_mtime_ns} for p in documents]
        return manifest.get("files") == current
    except (OSError, json.JSONDecodeError):
        return False


class AppState:
    def __init__(self, chat_id: str | None = None):
        self.chat_id = chat_id or next(iter(registry()), None) or create_chat()
        self.lock = threading.Lock()
        self.build_lock = threading.Lock()
        self.build = {"running": False, "ready": False, "phase": "idle", "progress": 0, "message": ""}
        self.bot: QABot | None = None
        self.refresh()

    @property
    def paths(self) -> dict[str, Path]:
        return paths_for(self.chat_id)

    def refresh(self) -> None:
        paths = self.paths
        self.build["ready"] = paths["vector"].exists() and matching_index(paths)
        self.bot = QABot(str(paths["vector"]), str(paths["bm25"]), str(paths["history"]))

    def switch(self, chat_id: str) -> None:
        if chat_id not in registry():
            raise ValueError("Không tìm thấy cuộc trò chuyện.")
        self.chat_id = chat_id
        self.build = {"running": False, "ready": False, "phase": "idle", "progress": 0, "message": ""}
        self.refresh()

    def answer(self, question: str) -> tuple[str, str]:
        if self.build["running"] or not self.build["ready"] or self.bot is None:
            raise RuntimeError("Index của cuộc trò chuyện chưa sẵn sàng. Hãy import và build tài liệu.")
        with self.lock:
            return self.bot.answer(question)

    def start_build(self, start_page: int, page_count: int) -> None:
        with self.build_lock:
            if self.build["running"]:
                raise RuntimeError("Đang có một tiến trình build.")
            self.build.update(running=True, ready=False, phase="starting", progress=0, message="Đang chuẩn bị...")
        threading.Thread(target=self._build_worker, args=(start_page, page_count), daemon=True).start()

    def _build_worker(self, start_page: int, page_count: int) -> None:
        started = time.monotonic()
        paths = self.paths

        def progress(payload: dict) -> None:
            phase = payload.get("phase", "loading")
            value = {"loading": 12, "page": 58, "embedding": 78, "done": 100}.get(phase, 12)
            self.build.update(phase=phase, progress=value, message=payload.get("message", payload.get("file", "")))

        try:
            result = build_index(start_page, page_count, progress=progress, data_path=str(paths["documents"]),
                                 vector_db_path=str(paths["vector"]), bm25_index_path=str(paths["bm25"]),
                                 manifest_path=str(paths["manifest"]))
            elapsed = int(time.monotonic() - started)
            self.build.update(running=False, ready=True, phase="done", progress=100,
                              message=f"Hoàn tất trong {elapsed}s", result=result)
            self.bot = QABot(str(paths["vector"]), str(paths["bm25"]), str(paths["history"]))
        except Exception as error:
            self.build.update(running=False, ready=False, phase="error", progress=0, message=str(error))


STATE = AppState()


def respond(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def status_payload() -> dict:
    paths = STATE.paths
    documents = sorted(paths["documents"].glob("*.pdf")) if paths["documents"].exists() else []
    chats = registry()
    return {"chat_id": STATE.chat_id, "title": chats.get(STATE.chat_id, {}).get("title", "Cuộc trò chuyện"),
            "ready": STATE.build["ready"] and matching_index(paths) and not STATE.build["running"],
            "build": dict(STATE.build), "pdf_count": len(documents), "pdfs":[p.name for p in documents],
            "model": OLLAMA_MODEL, "history_count": len(load_history(str(paths["history"]))) }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/status":
            respond(self, status_payload()); return
        if route == "/api/chats":
            data = registry()
            respond(self, {"active": STATE.chat_id, "chats": [{"id": k, **v} for k, v in data.items()]}); return
        if route == "/api/history":
            respond(self, {"history": load_history(str(STATE.paths["history"]))}); return
        files = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"),
                 "/styles.css": ("styles.css", "text/css; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8")}
        if route in files:
            name, content_type = files[route]; self.serve_file(WEB_ROOT / name, content_type); return
        self.send_error(404)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/upload":
            content_type = self.headers.get("Content-Type", "")
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD":"POST", "CONTENT_TYPE":content_type})
            item = form["file"] if "file" in form else None
            filename = Path(getattr(item, "filename", "")).name if item is not None else ""
            if not filename.lower().endswith(".pdf"):
                respond(self, {"error":"Chỉ hỗ trợ file PDF."}, 400); return
            destination = STATE.paths["documents"] / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as output: output.write(item.file.read())
            data = registry()
            if data.get(STATE.chat_id, {}).get("title") == "Cuộc trò chuyện mới":
                data[STATE.chat_id]["title"] = filename
                save_registry(data)
            STATE.build.update(ready=False, phase="idle", progress=0, message="Đã thêm tài liệu; cần build lại.")
            respond(self, {"ok":True, "filename":filename}); return

        length = int(self.headers.get("Content-Length", "0"))
        try: payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError: respond(self, {"error":"Dữ liệu không hợp lệ."}, 400); return

        try:
            if route == "/api/chat":
                answer, citation = STATE.answer(str(payload.get("question", "")).strip())
                respond(self, {"answer":answer, "citation":citation}); return
            if route == "/api/build":
                STATE.start_build(max(1, int(payload.get("start_page", 1))), max(0, int(payload.get("page_count", 0))))
                respond(self, {"ok":True}); return
            if route == "/api/clear":
                clear_history(str(STATE.paths["history"])); STATE.refresh(); respond(self, {"ok":True}); return
            if route == "/api/new-chat":
                chat_id = create_chat(); STATE.switch(chat_id); respond(self, status_payload()); return
            if route == "/api/switch-chat":
                STATE.switch(str(payload.get("chat_id", ""))); respond(self, status_payload()); return
            if route == "/api/delete-document":
                filename = Path(str(payload.get("filename", ""))).name
                target = STATE.paths["documents"] / filename
                if not target.exists(): respond(self, {"error":"Không tìm thấy tài liệu."}, 404); return
                target.unlink(); STATE.build.update(ready=False, phase="idle", progress=0, message="Đã xóa tài liệu; cần build lại.")
                respond(self, {"ok":True}); return
            if route == "/api/clear-documents":
                paths = STATE.paths
                for document in paths["documents"].glob("*.pdf"):
                    document.unlink()
                shutil.rmtree(paths["vector"], ignore_errors=True)
                STATE.build.update(ready=False, phase="idle", progress=0, message="Đã xóa toàn bộ tài liệu và index.")
                STATE.bot = QABot(str(paths["vector"]), str(paths["bm25"]), str(paths["history"]))
                respond(self, {"ok":True}); return
            if route == "/api/delete-chat":
                chat_id = str(payload.get("chat_id", ""))
                if chat_id == STATE.chat_id: raise ValueError("Không thể xóa cuộc trò chuyện đang mở.")
                paths = paths_for(chat_id); shutil.rmtree(paths["root"], ignore_errors=True)
                data = registry(); data.pop(chat_id, None); save_registry(data)
                respond(self, {"ok":True}); return
        except (RuntimeError, ValueError, FileNotFoundError) as error:
            respond(self, {"error":str(error)}, 400); return
        self.send_error(404)

    def serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists(): self.send_error(404); return
        body = path.read_bytes(); self.send_response(200); self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None: print(f"[web] {format % args}")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("QA Bot web UI: http://127.0.0.1:8000")
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nĐã dừng web UI.")
    finally: server.server_close()


if __name__ == "__main__": main()
