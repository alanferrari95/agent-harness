from __future__ import annotations

from conftest import FIX_ADD, ScriptedModel, make_config

from agent_harness.loop import FINISH_REJECTED, Harness
from agent_harness.models.base import ContextOverflow, ModelError, ToolCall
from agent_harness.tracing import read_trace

RUN_TESTS = ToolCall("run_tests", {})
FINISH = ToolCall("finish", {"summary": "done"})


def test_happy_path_is_verified_by_the_harness(task, tmp_path):
    trace = tmp_path / "trace.jsonl"
    result = Harness(make_config(), ScriptedModel([FIX_ADD, RUN_TESTS, FINISH])).run(task, trace)
    assert result.success and result.stop_reason == "finished" and result.steps == 3
    events = read_trace(trace)
    names = [e["event"] for e in events]
    assert names[0] == "run_start" and names[-1] == "run_end"
    assert names.count("step") == 3


def test_trace_records_the_exact_prompt(task, tmp_path):
    trace = tmp_path / "trace.jsonl"
    model = ScriptedModel([FIX_ADD, RUN_TESTS, FINISH])
    Harness(make_config(), model).run(task, trace)
    steps = [e for e in read_trace(trace) if e["event"] == "step"]
    assert [s["prompt"] for s in steps] == [r.prompt for r in model.requests]


def test_finish_is_a_claim_not_evidence(task):
    result = Harness(make_config(), ScriptedModel([FINISH])).run(task)
    assert result.stop_reason == "finished" and not result.success


def test_step_budget(task):
    model = ScriptedModel([ToolCall("search_code", {"query": str(i)}) for i in range(20)])
    result = Harness(make_config(loop={"max_steps": 4}), model).run(task)
    assert result.stop_reason == "budget_steps" and result.steps == 4


def test_cost_budget(task):
    calls = [ToolCall("search_code", {"query": str(i)}) for i in range(20)]
    model = ScriptedModel(calls, cost=0.3)
    result = Harness(make_config(loop={"max_cost_usd": 1.0}), model).run(task)
    assert result.stop_reason == "budget_cost"
    assert result.steps == 4 and result.cost_usd == 1.2


def test_time_budget(task):
    result = Harness(make_config(loop={"max_seconds": 0}), ScriptedModel([FIX_ADD])).run(task)
    assert result.stop_reason == "budget_time" and result.steps == 0


def test_stuck_detection_stops_on_third_identical_call(task):
    same = ToolCall("search_code", {"query": "add"})
    result = Harness(make_config(loop={"stuck_threshold": 3}), ScriptedModel([same] * 10)).run(task)
    assert result.stop_reason == "stuck" and result.steps == 3


def test_stuck_detection_off_by_default(task):
    same = ToolCall("search_code", {"query": "add"})
    result = Harness(make_config(loop={"max_steps": 5}), ScriptedModel([same] * 10)).run(task)
    assert result.stop_reason == "budget_steps"


def test_verify_on_stop_rejects_untested_edits(task):
    model = ScriptedModel([FIX_ADD, FINISH, RUN_TESTS, FINISH])
    result = Harness(make_config(loop={"verify_on_stop": True}), model).run(task)
    assert result.success and result.steps == 4
    assert FINISH_REJECTED in model.requests[2].prompt


def test_malformed_replies_are_capped(task):
    result = Harness(make_config(loop={"max_malformed": 2}), ScriptedModel([None, None, FIX_ADD])).run(task)
    assert result.stop_reason == "malformed" and result.steps == 2


def test_unknown_tool_goes_back_as_an_observation(task):
    model = ScriptedModel([ToolCall("delete_repo", {}), FINISH])
    Harness(make_config(), model).run(task)
    assert "unknown tool 'delete_repo'" in model.requests[1].prompt


def test_undeclared_tools_are_not_shown_to_the_model(task):
    rules = [{"tool": "read_file", "decision": "allow"}]
    model = ScriptedModel([FINISH])
    Harness(make_config(rules=rules), model).run(task)
    assert [t.name for t in model.requests[0].tools] == ["read_file", "finish"]


def test_deny_rule_blocks_edits_to_tests(task):
    rules = [
        {"tool": "edit_file", "paths": ["tests/*"], "decision": "deny"},
        {"tool": "edit_file", "decision": "allow"},
    ]
    cheat = ToolCall(
        "edit_file",
        {"path": "tests/test_calc.py", "old_text": "add(2, 3), 5", "new_text": "add(2, 3), -1"},
    )
    model = ScriptedModel([cheat, FINISH])
    result = Harness(make_config(rules=rules), model).run(task)
    assert "DENIED by permission policy" in model.requests[1].prompt
    assert result.verification["tampered_files"] == []


def test_ask_rule_consults_the_approver(task):
    rules = [{"tool": "edit_file", "decision": "ask"}]
    seen = []

    def refuse(tool, args):
        seen.append(tool)
        return False

    model = ScriptedModel([FIX_ADD, FINISH])
    result = Harness(make_config(rules=rules), model, approver=refuse).run(task)
    assert seen == ["edit_file"] and not result.success
    assert "DENIED: the user did not approve" in model.requests[1].prompt


def test_tampered_tests_do_not_fool_verification(task):
    cheat = ToolCall(
        "edit_file",
        {"path": "tests/test_calc.py", "old_text": "add(2, 3), 5", "new_text": "add(2, 3), -1"},
    )
    result = Harness(make_config(), ScriptedModel([cheat, RUN_TESTS, FINISH])).run(task)
    assert not result.success
    assert result.verification["tampered_files"] == ["tests/test_calc.py"]


def test_context_overflow_and_model_errors_end_the_run_cleanly(task):
    def overflow(request):
        raise ContextOverflow("too long")

    def broken(request):
        raise ModelError("HTTP 500")

    assert Harness(make_config(), ScriptedModel([overflow])).run(task).stop_reason == "context_overflow"
    assert Harness(make_config(), ScriptedModel([broken])).run(task).stop_reason == "model_error"


def test_state_lives_outside_the_model(task):
    model = ScriptedModel([ToolCall("read_file", {"path": "calc.py"}), FIX_ADD, RUN_TESTS, FINISH])
    result = Harness(make_config(), model).run(task)
    assert result.state["files_read"] == ["calc.py"]
    assert result.state["files_edited"] == ["calc.py"]
    assert result.state["last_test"] == "PASSED" and result.state["last_error"] is None
    assert '"files_read": ["calc.py"]' in model.requests[2].prompt


def test_the_original_repository_is_untouched(task):
    before = {p: p.read_bytes() for p in task.repo.rglob("*") if p.is_file()}
    Harness(make_config(), ScriptedModel([FIX_ADD, RUN_TESTS, FINISH])).run(task)
    after = {p: p.read_bytes() for p in task.repo.rglob("*") if p.is_file()}
    assert before == after
