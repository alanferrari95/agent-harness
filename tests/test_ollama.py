"""Ollama adapter: request shape, parsing, the context guard and a real HTTP round trip
against a stand-in server (no Ollama install needed)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from conftest import make_config

from agent_harness.evals import model_slug
from agent_harness.loop import Harness
from agent_harness.models.base import ContextOverflow, ModelError, ModelRequest, ToolSpec
from agent_harness.models.factory import make_model
from agent_harness.models.ollama_model import OllamaModel

REQUEST = ModelRequest(
    "system prompt",
    "the prompt",
    [ToolSpec("read_file", "Read a file.", {"type": "object", "properties": {}, "required": []})],
)


def test_payload_shape():
    body = OllamaModel("ollama/qwen3:8b", num_ctx=8192, think=False).payload(REQUEST)
    assert body["model"] == "qwen3:8b" and body["stream"] is False
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["tools"][0] == {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    assert body["options"]["num_ctx"] == 8192 and body["think"] is False


def test_think_is_omitted_by_default():
    assert "think" not in OllamaModel("ollama/qwen3:8b").payload(REQUEST)


def test_parse_tool_call_and_usage():
    data = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}],
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 900,
        "eval_count": 40,
    }
    parsed = OllamaModel("ollama/qwen3:8b").parse(data)
    assert parsed.tool_call.name == "read_file" and parsed.tool_call.args == {"path": "a.py"}
    assert (parsed.input_tokens, parsed.output_tokens, parsed.cost_usd) == (900, 40, 0.0)
    assert parsed.tokens_estimated is False


def test_parse_string_arguments_and_plain_text():
    model = OllamaModel("ollama/qwen3:8b")
    data = {"message": {"tool_calls": [{"function": {"name": "x", "arguments": '{"q": 1}'}}]}}
    assert model.parse(data).tool_call.args == {"q": 1}
    text_only = model.parse({"message": {"content": "I think..."}, "done_reason": "length"})
    assert text_only.tool_call is None and "done_reason=length" in text_only.text


def test_context_guard_refuses_instead_of_truncating():
    model = OllamaModel("ollama/qwen3:8b", num_ctx=1000, max_tokens=100)
    big = ModelRequest("s", "x" * 8000, [])
    with pytest.raises(ContextOverflow):
        model.complete(big)


def test_unreachable_server_is_a_model_error(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    with pytest.raises(ModelError, match="cannot reach Ollama"):
        OllamaModel("ollama/qwen3:8b", timeout=2).complete(REQUEST)


def test_slug_is_directory_safe():
    assert model_slug("ollama/qwen3:8b") == "ollama_qwen3-8b"


@pytest.fixture
def fake_ollama(monkeypatch):
    """A tiny HTTP server that answers /api/chat like Ollama, from a script."""
    replies = []
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib name
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            call = replies.pop(0)
            body = {
                "message": {"role": "assistant", "content": "", "tool_calls": [{"function": call}]},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 500,
                "eval_count": 20,
            }
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{server.server_port}")
    yield replies, seen
    server.shutdown()


def test_full_run_over_http(task, fake_ollama):
    replies, seen = fake_ollama
    replies += [
        {"name": "edit_file", "arguments": {"path": "calc.py", "old_text": "return a - b",
                                            "new_text": "return a + b"}},
        {"name": "run_tests", "arguments": {}},
        {"name": "finish", "arguments": {"summary": "fixed"}},
    ]
    config = make_config()
    model = make_model("ollama/qwen3:8b", task, config.model)
    result = Harness(config, model).run(task)
    assert result.success and result.steps == 3
    assert result.input_tokens == 1500 and result.cost_usd == 0.0
    assert seen[0]["model"] == "qwen3:8b" and seen[0]["options"]["num_ctx"] == 16384
