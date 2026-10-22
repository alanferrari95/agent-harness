"""The model interface.

Any object with a `name` and a `complete(request)` method is a model. There is no base
class to inherit and no registry: structural typing, the lightest extension point the
paper found in its corpus (Mini-SWE-Agent's `Model` protocol).

One call returns at most one action. The harness rebuilds the whole prompt every step
from its own state, so a model never needs to remember anything between calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    """What the model sees of a tool: a name, a description and a JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRequest:
    system: str
    prompt: str
    tools: list[ToolSpec]


@dataclass
class ModelResponse:
    tool_call: ToolCall | None
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # True when token counts are a chars/4 estimate rather than provider-reported usage.
    tokens_estimated: bool = False


class ModelError(Exception):
    """The provider failed (network, auth, malformed payload). Ends the run."""


class ContextOverflow(ModelError):
    """The prompt does not fit in the model's context window."""


class Model(Protocol):
    name: str

    def complete(self, request: ModelRequest) -> ModelResponse: ...


def estimate_tokens(text: str) -> int:
    """Rough estimate, about four characters per token. Good for thresholds, not billing."""
    return len(text) // 4
