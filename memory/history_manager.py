import os, json
from datetime import datetime
from config import CHAT_HISTORY_PATH

def save_turn(user_msg: str, bot_msg: str, history_path: str = CHAT_HISTORY_PATH):
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    entry = {"user": user_msg, "bot": bot_msg,
             "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_history(history_path: str = CHAT_HISTORY_PATH):
    if not os.path.exists(history_path):
        return []
    history = []
    with open(history_path, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                history.append({"role": "user",      "content": e["user"]})
                history.append({"role": "assistant",  "content": e["bot"]})
            except Exception:
                pass
    return history

def clear_history(history_path: str = CHAT_HISTORY_PATH):
    if os.path.exists(history_path):
        os.remove(history_path)