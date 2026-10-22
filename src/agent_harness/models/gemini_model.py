"""Gemini adapter (Gemini API `models.generateContent`, plain REST, no SDK).

Reads the key from GEMINI_API_KEY and sends it in the `x-goog-api-key` header. Tools go
in `tools[].functionDeclarations[]` with `parametersJsonSchema`; the reply's
`functionCall` parts carry the action. As with the Claude adapter, each step is a
single user turn, so no thought signatures have to be replayed.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .base import ContextOverflow, ModelError, ModelRequest, ModelResponse, ToolCall
from .pricing import cost_usd

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiModel:
    def __init__(self, model_id: str, max_tokens: int = 4096, timeout: float = 120.0):
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise ModelError("GEMINI_API_KEY is not set; export it to use Gemini models.")
        self.name = model_id
        self.max_tokens = max_tokens
        self.timeout = timeout

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        return {
            "systemInstruction": {"parts": [{"text": request.system}]},
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parametersJsonSchema": t.parameters,
                        }
                        for t in request.tools
                    ]
                }
            ],
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }

    def complete(self, request: ModelRequest) -> ModelResponse:
        body = json.dumps(self.payload(request)).encode("utf-8")
        http_request = urllib.request.Request(
            ENDPOINT.format(model=self.name),
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as reply:
                data = json.loads(reply.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            if exc.code == 400 and "token" in detail.lower() and "exceed" in detail.lower():
                raise ContextOverflow(detail) from exc
            raise ModelError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ModelError(f"network error: {exc}") from exc
        return self.parse(data)

    def parse(self, data: dict[str, Any]) -> ModelResponse:
        call = None
        texts = []
        candidates = data.get("candidates") or []
        parts = (candidates[0].get("content") or {}).get("parts", []) if candidates else []
        for part in parts:
            if "functionCall" in part and call is None:
                fc = part["functionCall"]
                call = ToolCall(fc["name"], dict(fc.get("args") or {}))
            elif "text" in part and not part.get("thought"):
                texts.append(part["text"])
        if candidates and candidates[0].get("finishReason") not in (None, "STOP"):
            texts.append(f"[finishReason={candidates[0]['finishReason']}]")
        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        # Thinking tokens are billed as output.
        output_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        return ModelResponse(
            tool_call=call,
            text="\n".join(texts),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(self.name, input_tokens, output_tokens),
        )
