"""Local models through Ollama (`POST /api/chat`, plain REST, no SDK).

No key and no cost: the model runs on this machine. The model ID is written
`ollama/<name>`, e.g. `ollama/qwen3:8b`. The server address comes from OLLAMA_HOST
(default http://localhost:11434).

Two details matter for a harness:
- `num_ctx` is always sent. Ollama's documented default is small (2048 tokens) and the
  docs do not say what happens to a longer prompt, so the adapter also refuses, before
  sending, any prompt whose estimate exceeds it: a ContextOverflow instead of a silent
  truncation.
- Token counts are the server's own (`prompt_eval_count`, `eval_count`); cost is 0.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .base import (
    ContextOverflow,
    ModelError,
    ModelRequest,
    ModelResponse,
    ToolCall,
    estimate_tokens,
)

PREFIX = "ollama/"
DEFAULT_HOST = "http://localhost:11434"


class OllamaModel:
    def __init__(
        self,
        model_id: str,
        num_ctx: int = 16384,
        max_tokens: int = 4096,
        think: bool | None = None,
        timeout: float = 600.0,
    ):
        self.name = model_id
        self.local_name = model_id.removeprefix(PREFIX)
        self.host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
        if not self.host.startswith("http"):
            self.host = f"http://{self.host}"
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.think = think
        self.timeout = timeout

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.local_name,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {"num_ctx": self.num_ctx, "num_predict": self.max_tokens},
        }
        if self.think is not None:
            body["think"] = self.think
        return body

    def complete(self, request: ModelRequest) -> ModelResponse:
        body = self.payload(request)
        estimate = estimate_tokens(json.dumps(body))
        if estimate + self.max_tokens > self.num_ctx:
            raise ContextOverflow(
                f"prompt is ~{estimate} tokens plus {self.max_tokens} for the reply; "
                f"num_ctx is {self.num_ctx}"
            )
        http_request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as reply:
                data = json.loads(reply.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise ModelError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ModelError(
                f"cannot reach Ollama at {self.host} ({exc}). Is it running? "
                f"Try: ollama pull {self.local_name}"
            ) from exc
        return self.parse(data)

    def parse(self, data: dict[str, Any]) -> ModelResponse:
        message = data.get("message") or {}
        call = None
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            args = function.get("arguments") or {}
            if isinstance(args, str):  # some models emit a JSON string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_unparsed": args}
            call = ToolCall(function.get("name", ""), dict(args))
            break
        text = message.get("content") or ""
        if data.get("done_reason") not in (None, "stop"):
            text += f"\n[done_reason={data['done_reason']}]"
        return ModelResponse(
            tool_call=call,
            text=text.strip(),
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            cost_usd=0.0,
        )
