import time
import json
import threading
from typing import Any, Optional, Tuple
import re
import os
from datetime import datetime
import config
try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except Exception:
    _HAS_JSONSCHEMA = False
try:
    from utils import session_store
except Exception:
    session_store = None

# Lightweight LLM client wrapper supporting Tongyi, OpenAI, or local fallback.
class LLMClient:
    def __init__(self, backend: str = "auto", openai_api_key: str = "", dashscope_key: str = "", model_name: str = "qwen-turbo", temperature: float = 0.0, retries: int = 2, backoff: float = 1.0):
        self.backend = backend
        self.openai_api_key = openai_api_key
        self.dashscope_key = dashscope_key
        self.model_name = model_name
        self.temperature = temperature
        self.retries = retries
        self.backoff = backoff
        self.lock = threading.Lock()
        # Token bucket for rate limiting
        self.capacity = float(config.RATE_LIMIT_PER_SECOND)
        self.tokens = self.capacity
        self.fill_rate = float(config.RATE_LIMIT_PER_SECOND)
        self.timestamp = time.time()
        # simple call/cost logging
        self.cost_log = config.COST_LOG_PATH

    def _call_tongyi(self, prompt: str) -> str:
        try:
            from langchain_community.llms import Tongyi
        except Exception as exc:
            raise RuntimeError("Tongyi client library not available: %s" % exc)
        model = Tongyi(model_name=self.model_name, temperature=self.temperature, api_key=self.dashscope_key)
        # The Tongyi wrapper returns a chat-like object; call it like a function if supported
        return model(prompt)

    def _call_openai(self, prompt: str) -> str:
        try:
            import openai
        except Exception as exc:
            raise RuntimeError("OpenAI package not available: %s" % exc)
        openai.api_key = self.openai_api_key
        resp = openai.ChatCompletion.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=self.temperature)
        return resp

    def _call_local(self, prompt: str) -> Tuple[str, dict]:
        # Very simple local fallback: return a simple structured JSON when schema requested is simple
        text = "[LOCAL_FALLBACK]\n" + (prompt[:1000])
        return text, {}

    def _extract_json_from_text(self, text: str) -> Optional[Any]:
        # More robust JSON extraction: find first { or [ and parse until matching bracket,
        # ignoring brackets inside string literals.
        if not isinstance(text, str):
            return None
        start_idx = None
        for i, ch in enumerate(text):
            if ch == '{' or ch == '[':
                start_idx = i
                break
        if start_idx is None:
            return None
        stack = []
        in_str = False
        escape = False
        for i in range(start_idx, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == '{' or ch == '[':
                    stack.append(ch)
                elif ch == '}' or ch == ']':
                    if not stack:
                        # unmatched closing
                        return None
                    opening = stack.pop()
                    if (opening == '{' and ch != '}') or (opening == '[' and ch != ']'):
                        return None
                    if not stack:
                        # matched full JSON
                        candidate = text[start_idx:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            return None
        return None

    def _call_backend(self, prompt: str) -> Tuple[str, dict]:
        """Call configured backend and return (text, metadata) where metadata may include token usage."""
        if self.backend == "tongyi":
            text = self._call_tongyi(prompt)
            return text, {}
        elif self.backend == "openai":
            resp = self._call_openai(prompt)
            # try extracting text and usage
            try:
                text = resp["choices"][0]["message"]["content"]
            except Exception:
                text = str(resp)
            usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
            return text, usage
        else:
            # local
            text, meta = self._call_local(prompt)
            return text, meta

    def send(self, prompt: str, structured_schema: Optional[dict] = None) -> Any:
        """Send prompt to configured backend with retry/backoff and schema validation.
        If structured_schema provided and jsonschema available, enforce JSON response and validate.
        Returns parsed JSON on success, otherwise raw text.
        """
        if structured_schema is not None and not _HAS_JSONSCHEMA:
            # warn in log but proceed; schema validation unavailable
            try:
                with open(self.cost_log, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.utcnow().isoformat()}Z\tWARNING jsonschema not installed, schema validation skipped\n")
            except Exception:
                pass

        last_err = None
        # augment prompt with schema instruction when schema provided
        base_prompt = prompt
        if structured_schema is not None:
            try:
                schema_text = json.dumps(structured_schema, ensure_ascii=False)
                base_prompt = (
                    prompt + "\n\n请仅以 JSON 格式响应，严格遵循下面的 JSON Schema（不要添加其他说明文本）：\n" + schema_text
                )
            except Exception:
                base_prompt = prompt

        for attempt in range(self.retries + 1):
            try:
                # rate limit refill
                now = time.time()
                elapsed = now - self.timestamp
                self.timestamp = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
                if self.tokens < 1.0:
                    wait = (1.0 - self.tokens) / self.fill_rate
                    time.sleep(wait)
                    self.tokens = min(self.capacity, self.tokens + wait * self.fill_rate)
                self.tokens -= 1.0

                start = time.time()
                text, meta = self._call_backend(base_prompt)
                duration = time.time() - start

                # log call with metadata
                try:
                    tokens_info = meta if isinstance(meta, dict) else {}
                    self._log_call(base_prompt, text, duration=duration, tokens_info=tokens_info)
                except Exception:
                    pass

                # If schema requested, try to extract and validate
                if structured_schema is not None:
                    parsed = self._extract_json_from_text(text)
                    if parsed is not None:
                        if _HAS_JSONSCHEMA:
                            try:
                                jsonschema.validate(instance=parsed, schema=structured_schema)
                                return parsed
                            except Exception as v_err:
                                # validation failed; record and retry
                                last_err = v_err
                                self._audit_schema_failure(
                                    prompt=base_prompt,
                                    schema=structured_schema,
                                    raw_output=text,
                                    error=str(v_err),
                                    attempt=attempt,
                                    repair_instruction="请仅返回符合给定 JSON Schema 的 JSON，不要附加任何说明。"
                                )
                                base_prompt = (
                                    "上次返回的 JSON 未通过验证，请仅返回符合给定 JSON Schema 的 JSON，不要附加任何说明。验证错误：" + str(v_err) + "\n" + json.dumps(structured_schema, ensure_ascii=False)
                                )
                                time.sleep(self.backoff * (2 ** attempt))
                                continue
                        else:
                            return parsed
                    else:
                        # couldn't extract JSON; retry
                        last_err = RuntimeError("无法从模型输出中提取 JSON")
                        self._audit_schema_failure(
                            prompt=base_prompt,
                            schema=structured_schema,
                            raw_output=text,
                            error="无法从模型输出中提取 JSON",
                            attempt=attempt,
                            repair_instruction="请仅返回符合给定 JSON Schema 的 JSON，不要添加任何多余文本。"
                        )
                        base_prompt = (
                            "请仅返回符合给定 JSON Schema 的 JSON，不要添加任何多余文本。模式：" + json.dumps(structured_schema, ensure_ascii=False)
                        )
                        time.sleep(self.backoff * (2 ** attempt))
                        continue

                # no schema requested: return raw text
                return text
            except Exception as e:
                last_err = e
                time.sleep(self.backoff * (2 ** attempt))

        # all attempts exhausted
        # fallback: if schema required, try local fallback for best-effort
        if structured_schema is not None:
            try:
                text, _ = self._call_local(prompt)
                parsed = self._extract_json_from_text(text)
                if parsed is not None:
                    return parsed
            except Exception:
                pass
        raise RuntimeError(f"LLM request failed after {self.retries+1} attempts: {last_err}")

    def _audit_schema_failure(self, prompt: str, schema: dict, raw_output: str, error: str, attempt: int, repair_instruction: str) -> None:
        if session_store is None:
            return
        try:
            session_store.append_session({
                "event_type": "schema_validation_failure",
                "backend": self.backend,
                "model": self.model_name,
                "attempt": attempt,
                "error": error,
                "prompt": prompt,
                "repair_instruction": repair_instruction,
                "raw_output": raw_output,
                "schema": schema,
            })
        except Exception:
            pass

    def _log_call(self, prompt: str, response: str, duration: float = 0.0, tokens_info: dict | None = None) -> None:
        # extended logging: timestamp, backend, prompt len, response len, duration, token usage if available
        try:
            os.makedirs(os.path.dirname(self.cost_log), exist_ok=True)
            with open(self.cost_log, 'a', encoding='utf-8') as f:
                line = f"{datetime.utcnow().isoformat()}Z\tbackend={self.backend}\tmodel={self.model_name}\tprompt_len={len(prompt)}\tresp_len={len(str(response))}\tduration={duration:.3f}"
                if tokens_info:
                    try:
                        line += "\tprompt_tokens=%s\tcompletion_tokens=%s\ttotal_tokens=%s" % (
                            tokens_info.get('prompt_tokens', ''), tokens_info.get('completion_tokens', ''), tokens_info.get('total_tokens', '')
                        )
                    except Exception:
                        pass
                line += "\n"
                f.write(line)
        except Exception:
            pass
