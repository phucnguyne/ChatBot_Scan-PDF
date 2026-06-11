import subprocess, sys, os

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

def check_ollama_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=3)
        return True
    except Exception:
        return False

def pull_model(model: str) -> None:
    print(f"Pulling Ollama model: {model} ...")
    result = subprocess.run(["ollama", "pull", model], capture_output=False)
    if result.returncode != 0:
        print(f"❌ ollama pull thất bại. Hãy chạy thủ công: ollama pull {model}")
    else:
        print(f"✅ Model {model} sẵn sàng.")

if __name__ == "__main__":
    if not check_ollama_running():
        print("⚠️  Ollama chưa chạy. Khởi động bằng: ollama serve")
        sys.exit(1)
    pull_model(OLLAMA_MODEL)