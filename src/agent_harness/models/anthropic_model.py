"""Claude adapter (Anthropic Messages API, official `anthropic` SDK).

Reads the key from ANTHROPIC_API_KEY. Design choices:
- One user message per step, rebuilt by the harness. No assistant turns are replayed,
  so there are no thinking blocks or tool results to keep byte-identical across calls.
  The cost: only the stable prefix (tools + system prompt) is cached, not the history.
- `cache_control` on the system block: tools and system never change within a run.
- `tool_choice` is `auto` with parallel calls disabled (one action per step). Forced
  tool choice returns a 400 on current Opus/Sonnet models, so it is never sent.
- No `temperature`/`top_p`/`top_k`: rejected by the API since 2026-10-21.
- Thinking stays at the model default; `effort` is configurable via output_config.
"""

from __future__ import annotations

import os
from typing import Any

from .base import ContextOverflow, ModelError, ModelRequest, ModelResponse, ToolCall
from .pricing import cost_usd


class AnthropicModel:
    def __init__(self, model_id: str, max_tokens: int = 4096, effort: str | None = None):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ModelError("ANTHROPIC_API_KEY is not set; export it to use Claude models.")
        try:
            import anthropic
        except ImportError as exc:
            raise ModelError("pip install 'agent-harness[anthropic]' to use Claude models.") from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.name = model_id
        self.max_tokens = max_tokens
        self.effort = effort

    def _params(self, request: ModelRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.name,
            "max_tokens": self.max_tokens,
            "system": [
                {"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}
            ],
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in request.tools
            ],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        return params

    def complete(self, request: ModelRequest) -> ModelResponse:
        anthropic = self._anthropic
        try:
            response = self.client.messages.create(**self._params(request))
        except anthropic.BadRequestError as exc:
            message = str(exc)
            if "context" in message.lower() and "window" in message.lower():
                raise ContextOverflow(message) from exc
            raise ModelError(f"bad request: {message}") from exc
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        return self.parse(response)

    def parse(self, response: Any) -> ModelResponse:
        call = None
        texts = []
        for block in response.content:
            if block.type == "tool_use" and call is None:
                call = ToolCall(block.name, dict(block.input or {}))
            elif block.type == "text":
                texts.append(block.text)
        if response.stop_reason == "refusal":
            texts.append("[model refused]")
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        input_tokens = usage.input_tokens + cache_read + cache_write
        return ModelResponse(
            tool_call=call,
            text="\n".join(texts),
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost_usd(
                self.name, usage.input_tokens, usage.output_tokens, cache_read, cache_write
            ),
        )
