import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.llm_utils import LLMClient

schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "top5": {
            "type": "array",
            "items": {"type": "object"}
        }
    },
    "required": ["summary"]
}

client = LLMClient(backend='local', retries=2, backoff=0.1)
# monkeypatch backend to simulate a model returning JSON
client._call_backend = lambda prompt: ('{\"summary\": \"测试通过\", \"top5\": [{\"id\":1}]}', {})
res = client.send('请返回 json', structured_schema=schema)
assert isinstance(res, dict) and res.get('summary') == '测试通过'
print('测试成功')
