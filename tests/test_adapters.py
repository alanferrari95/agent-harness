"""Provider adapters, tested against recorded response shapes (no network, no key)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from agent_harness.models.base import ModelError, ModelRequest, ToolSpec
from agent_harness.models.gemini_model import GeminiModel
from agent_harness.models.pricing import PRICES, cost_usd

REQUEST = ModelRequest(
    "system prompt",
    "the prompt",
    [ToolSpec("read_file", "Read a file.", {"type": "object", "properties": {}, "required": []})],
)


@pytest.fixture
def fake_anthropic(monkeypatch):
    """A stand-in `anthropic` module so the adapter can be built without the SDK."""
    module = types.ModuleType("anthropic")

    class _Error(Exception):
        pass

    module.BadRequestError = type("BadRequestError", (_Error,), {})
    module.APIStatusError = type("APIStatusError", (_Error,), {})
    module.APIConnectionError = type("APIConnectionError", (_Error,), {})
    module.Anthropic = lambda: SimpleNamespace(messages=SimpleNamespace(create=None))
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from agent_harness.models.anthropic_model import AnthropicModel

    return AnthropicModel


def test_anthropic_request_shape(fake_anthropic):
    model = fake_anthropic("claude-haiku-5-5", effort="low")
    params = model._params(REQUEST)
    assert params["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["tools"][0]["input_schema"]["type"] == "object"
    assert params["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(params)


def test_anthropic_response_parsing(fake_anthropic):
    model = fake_anthropic("claude-haiku-5-5")
    response = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Reading the file."),
            SimpleNamespace(type="tool_use", name="read_file", input={"path": "a.py"}, id="t1"),
            SimpleNamespace(type="tool_use", name="run_tests", input={}, id="t2"),
        ],
        usage=SimpleNamespace(
            input_tokens=1000, output_tokens=100, cache_read_input_tokens=500,
            cache_creation_input_tokens=0,
        ),
    )
    parsed = model.parse(response)
    assert parsed.tool_call.name == "read_file" and parsed.tool_call.args == {"path": "a.py"}
    assert parsed.text == "Reading the file."
    assert parsed.input_tokens == 1500 and parsed.output_tokens == 100
    assert parsed.cost_usd == pytest.approx(cost_usd("claude-haiku-5-5", 1000, 100, 500))


def test_anthropic_requires_a_key(fake_anthropic, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ModelError, match="ANTHROPIC_API_KEY"):
        fake_anthropic("claude-haiku-5-5")


def test_gemini_payload_and_parsing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    model = GeminiModel("gemini-3.8-flash")
    payload = model.payload(REQUEST)
    declaration = payload["tools"][0]["functionDeclarations"][0]
    assert declaration["parametersJsonSchema"]["type"] == "object"
    assert payload["systemInstruction"]["parts"][0]["text"] == "system prompt"
    reply = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "thinking...", "thought": True},
                        {"functionCall": {"name": "read_file", "args": {"path": "a.py"}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 800, "candidatesTokenCount": 20, "thoughtsTokenCount": 30},
    }
    parsed = model.parse(reply)
    assert parsed.tool_call.name == "read_file" and parsed.text == ""
    assert parsed.output_tokens == 50
    assert parsed.cost_usd == pytest.approx((800 * 0.75 + 50 * 3.75) / 1e6)


def test_gemini_requires_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ModelError, match="GEMINI_API_KEY"):
        GeminiModel("gemini-3.8-flash")


def test_every_priced_model_has_positive_prices():
    for model_id, price in PRICES.items():
        assert price.input_per_mtok > 0 and price.output_per_mtok > 0, model_id
    with pytest.raises(KeyError):
        cost_usd("unknown-model", 1, 1)
