from __future__ import annotations

import pytest

from agent_harness.models.base import ContextOverflow, ModelRequest
from agent_harness.models.fake import FakeModel


def prompt(observation: str) -> ModelRequest:
    text = f"# Task\nx\n\n# Last observation\nstep 1: something()\n{observation}"
    return ModelRequest("system", text, [])


EDIT = {
    "tool": "edit_file",
    "typo": True,
    "args": {"path": "a.py", "old_text": "    x = 1", "new_text": "    x = 2"},
}


def test_follows_its_script_then_finishes():
    model = FakeModel({"steps": [{"tool": "run_tests"}]})
    assert model.complete(prompt("")).tool_call.name == "run_tests"
    assert model.complete(prompt("FAILED")).tool_call.name == "finish"


def test_typo_recovers_only_from_a_quoted_hint():
    model = FakeModel({"steps": [EDIT]})
    first = model.complete(prompt("")).tool_call
    assert first.args["old_text"] == "    x  = 1"
    hint = 'ERROR: old_text not found in a.py. Closest line (line 3): "    x = 1". Copy the exact text'
    retry = model.complete(prompt(hint)).tool_call
    assert retry.name == "edit_file" and retry.args["old_text"] == "    x = 1"


def test_plain_error_gives_no_recovery():
    model = FakeModel({"steps": [EDIT]})
    model.complete(prompt(""))
    assert model.complete(prompt("ERROR: edit_file failed.")).tool_call.name == "finish"


def test_rejected_finish_leads_to_running_tests():
    model = FakeModel({"steps": [{"tool": "finish"}, {"tool": "edit_file", "args": {}}]})
    model.complete(prompt(""))
    assert model.complete(prompt("HARNESS: finish rejected. ...")).tool_call.name == "run_tests"
    assert model.complete(prompt("FAILED (exit code 1)")).tool_call.name == "edit_file"


def test_passing_tests_lead_to_finish():
    model = FakeModel({"steps": [{"tool": "run_tests"}, {"tool": "read_file", "args": {}}]})
    model.complete(prompt(""))
    assert model.complete(prompt("PASSED\nok")).tool_call.name == "finish"


def test_text_reply_without_tool_call():
    model = FakeModel({"steps": [{"tool": "none", "text": "hmm"}]})
    response = model.complete(prompt(""))
    assert response.tool_call is None and response.text == "hmm"


def test_context_window_is_enforced():
    model = FakeModel({"steps": []}, context_window_tokens=10)
    with pytest.raises(ContextOverflow):
        model.complete(prompt("x" * 200))
