"""httpx-based LLM helper — no openai SDK dependency.

Uses only httpx (pre-installed on Bailian platform) to call any
OpenAI-compatible chat completions endpoint.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx


def load_dotenv(path: str | None = None) -> None:
    """Load ``KEY=VALUE`` pairs from a .env file into os.environ.

    Searches (in order) an explicit ``path``, then ``.env`` in the current
    working directory, the repo root (parent of this module), and this module's
    directory. Never overrides a variable already set in the environment, so
    the Bailian platform's injected ``DASHSCOPE_*`` vars always win.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    candidates = []
    if path:
        candidates.append(path)
    candidates += [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(root, ".env"),
        os.path.join(here, ".env"),
    ]
    seen = set()
    for cand in candidates:
        cand = os.path.abspath(cand)
        if cand in seen or not os.path.isfile(cand):
            continue
        seen.add(cand)
        try:
            with open(cand, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip()
                    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                        val = val[1:-1]
                    if key and os.getenv(key) is None:
                        os.environ[key] = val
        except Exception:
            pass


load_dotenv()  # populate os.environ from .env before config is read


def _read_flat_config() -> dict[str, Any]:
    """Read config.yml as flat key/value pairs without pyyaml."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    result: dict[str, Any] = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                if value.lower() == "true":
                    result[key] = True
                elif value.lower() == "false":
                    result[key] = False
                elif value.isdigit():
                    result[key] = int(value)
                else:
                    result[key] = value
    except Exception:
        pass
    return result


_config = _read_flat_config()


def _cfg(key: str, default: str = "") -> str:
    env_val = os.getenv(key)
    if env_val is not None:
        return env_val
    val = _config.get(key, default)
    return str(val) if val is not None else default


def get_config(key: str, default: str = "") -> str:
    """Public config getter: environment variable wins over config.yml."""
    return _cfg(key, default)


def _strip_openai_suffix(url: str) -> str:
    value = (url or "").strip()
    for suffix in ("/chat/completions", "/completions", "/responses", "/embeddings"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value.rstrip("/")


async def chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    enable_search: bool = False,
) -> str:
    """Make an OpenAI-compatible chat completion request via httpx.

    When ``enable_search=True`` the request is sent with DashScope's
    ``extra_body["enable_search": true]`` which activates the model's built-in
    web-search capability (no extra API key needed — only the DashScope key is
    required, which the platform injects automatically).
    """
    api_key = _cfg("DASHSCOPE_API_KEY")
    if not api_key:
        return "[未配置 API Key，无法调用 LLM]"

    model = _cfg("DASHSCOPE_MODEL_CODE", "qwen-plus")
    api_url = _strip_openai_suffix(
        _cfg("DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    url = f"{api_url}/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if enable_search:
        # DashScope built-in web search — no extra API key needed
        payload["extra_body"] = {"enable_search": True}

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if not resp.is_success:
            return f"[LLM 调用失败: HTTP {resp.status_code}]"

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return f"[LLM 响应格式异常: {str(data)[:200]}]"


class LlmBudget:
    """Tracks the LLM call budget for the tutor dispatcher.

    Each sub-agent / planner / synthesizer call that hits an LLM endpoint
    must charge this counter so the whole tutoring turn stays within budget.
    """

    def __init__(self, max_calls: int) -> None:
        self.max_calls = int(max_calls)
        self.used = 0

    def can_charge(self, n: int = 1) -> bool:
        return self.used + n <= self.max_calls

    def charge(self, n: int = 1) -> None:
        self.used += n

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_calls


def normalize_json_output(raw_text: str | None) -> Any:
    """Strip markdown fences and parse JSON; return an error dict on failure."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "tool_error": "invalid_json",
            "message": f"无法解析 LLM 输出为 JSON: {exc.msg}",
            "raw_output": text[:2000],
        }


async def chat_completion_json(
    messages: list[dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> Any:
    """Call chat_completion and parse the result as JSON (dict or list)."""
    raw = await chat_completion(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return normalize_json_output(raw)
