import os
from huggingface_hub import hf_hub_download

os.makedirs("models", exist_ok=True)

hf_hub_download(
    repo_id="vilm/vinallama-7b-chat-GGUF",
    filename="vinallama-7b-chat_q5_0.gguf",
    local_dir="models"
)
print("Tải xong!")