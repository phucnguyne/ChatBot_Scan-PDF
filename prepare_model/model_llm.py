import subprocess, sys, os

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

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
        print(f"❌ ollama pull failed. Run this manually: ollama pull {model}")
    else:
        print(f"✅ Model {model} is ready.")

if __name__ == "__main__":
    if not check_ollama_running():
        print("⚠️  Ollama is not running. Start it with: ollama serve")
        sys.exit(1)
    pull_model(OLLAMA_MODEL)