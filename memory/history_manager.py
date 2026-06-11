import os, json
from datetime import datetime
from config import CHAT_HISTORY_PATH

def save_turn(user_msg: str, bot_msg: str):
    os.makedirs(os.path.dirname(CHAT_HISTORY_PATH), exist_ok=True)
    entry = {"user": user_msg, "bot": bot_msg,
             "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
    with open(CHAT_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_history():
    if not os.path.exists(CHAT_HISTORY_PATH):
        return []
    history = []
    with open(CHAT_HISTORY_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                history.append({"role": "user",      "content": e["user"]})
                history.append({"role": "assistant",  "content": e["bot"]})
            except Exception:
                pass
    return history

def clear_history():
    if os.path.exists(CHAT_HISTORY_PATH):
        os.remove(CHAT_HISTORY_PATH)