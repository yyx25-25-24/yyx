import json
import threading
from pathlib import Path
from typing import Any, Dict
import datetime
import config

_lock = threading.Lock()


def append_session(entry: Dict[str, Any]):
    p = Path(config.SESSION_STORE)
    with _lock:
        data = []
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding='utf-8') or '[]')
            except Exception:
                data = []
        entry.setdefault("timestamp", datetime.datetime.now().isoformat())
        data.append(entry)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def list_sessions() -> list:
    p = Path(config.SESSION_STORE)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding='utf-8') or '[]')
    except Exception:
        return []
