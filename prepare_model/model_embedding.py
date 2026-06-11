import os, shutil, urllib.request
from urllib.error import HTTPError, URLError

MODEL_DIR  = "models"
MODEL_FILE = "all-MiniLM-L6-v2-f16.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
MODEL_URL  = os.environ.get(
    "EMBEDDING_MODEL_URL",
    "https://gpt4all.io/models/gguf/all-MiniLM-L6-v2-f16.gguf"
)

os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(url: str, dest: str) -> None:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)

if not os.path.exists(MODEL_PATH):
    print(f"Đang tải embedding model (~46MB): {MODEL_FILE}")
    try:
        download_file(MODEL_URL, MODEL_PATH)
        print("✅ Tải xong!")
    except HTTPError as e:
        print(f"❌ HTTP {e.code} — thử đặt biến EMBEDDING_MODEL_URL hoặc tải thủ công.")
        raise
    except URLError:
        print("❌ Không kết nối được. Kiểm tra mạng hoặc đặt EMBEDDING_MODEL_URL.")
        raise
else:
    print(f"✅ Embedding model đã có: {MODEL_PATH}")