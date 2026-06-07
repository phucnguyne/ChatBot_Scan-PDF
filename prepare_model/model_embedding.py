import os
import shutil
import urllib.request
from urllib.error import HTTPError, URLError

model_dir = "models"
model_file = "all-MiniLM-L6-v2-f16.gguf"
model_path = os.path.join(model_dir, model_file)
default_model_url = "https://gpt4all.io/models/gguf/all-MiniLM-L6-v2-f16.gguf"
model_url = os.environ.get("MODEL_URL", default_model_url)

os.makedirs(model_dir, exist_ok=True)

def download_file(url: str, dest_path: str) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, open(dest_path, "wb") as file_obj:
        shutil.copyfileobj(response, file_obj)


if not os.path.exists(model_path):
    print("Đang tải model, vui lòng chờ (~46MB)...")
    try:
        download_file(model_url, model_path)
    except HTTPError as exc:
        if exc.code == 403:
            print("Loi 403. Hay thu dat bien MODEL_URL hoac tai thu cong.")
        raise
    except URLError:
        print("Khong the ket noi. Kiem tra internet hoac thu MODEL_URL.")
        raise
    print("Tải xong!")