import sys
import os
import json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.llm_utils import LLMClient
import config

# Use a dedicated temporary session file for test isolation
original_session_store = config.SESSION_STORE
config.SESSION_STORE = os.path.join(os.path.dirname(__file__), "session_history_test.json")
if os.path.exists(config.SESSION_STORE):
    os.remove(config.SESSION_STORE)

schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"}
    },
    "required": ["summary"]
}

responses = [
    ('{"summary": 123}', {}),
    ('{"summary": "修复成功"}', {})
]

client = LLMClient(backend='local', retries=2, backoff=0.1)

# simulate two backend responses: first invalid, then valid
client._call_backend = lambda prompt: responses.pop(0)

result = client.send('请返回 json', structured_schema=schema)
print('Result:', result)
assert isinstance(result, dict) and result['summary'] == '修复成功'

# verify audit log entry exists
history = []
with open(config.SESSION_STORE, 'r', encoding='utf-8') as f:
    history = json.loads(f.read())
assert any(entry.get('event_type') == 'schema_validation_failure' for entry in history)
print('Audit entry recorded')

# cleanup
try:
    os.remove(config.SESSION_STORE)
except Exception:
    pass
config.SESSION_STORE = original_session_store
