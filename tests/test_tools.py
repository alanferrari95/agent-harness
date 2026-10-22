from __future__ import annotations

import pytest

from agent_harness.tools import TOOLS, ToolContext, execute, validate_args
from agent_harness.workspace import Workspace


@pytest.fixture
def ctx(task):
    ws = Workspace.create(task.repo)
    yield ToolContext(ws, task.test_command, actionable_errors=True)
    ws.close()


def run(ctx, name, **args):
    return execute(TOOLS[name], args, ctx)


def test_edit_replaces_unique_match(ctx):
    out = run(ctx, "edit_file", path="calc.py", old_text="return a - b", new_text="return a + b")
    assert out.startswith("OK: edited calc.py")
    assert "return a + b" in (ctx.workspace.root / "calc.py").read_text()


def test_edit_not_found_quotes_closest_line(ctx):
    out = run(ctx, "edit_file", path="calc.py", old_text="return a  - b", new_text="x")
    assert out.startswith("ERROR: old_text not found in calc.py")
    assert 'Closest line (line 2): "    return a - b"' in out


def test_edit_ambiguous_reports_count_and_lines(ctx):
    out = run(ctx, "edit_file", path="calc.py", old_text="    return", new_text="    yield")
    assert "appears 2 times" in out
    assert "(lines 2, 6)" in out
    assert "Include more surrounding lines" in out


def test_edit_rejects_line_number_prefixes(ctx):
    out = run(ctx, "edit_file", path="calc.py", old_text="   2|     return a - b", new_text="x")
    assert "line-number prefixes" in out


def test_plain_errors_hide_the_hint(ctx):
    ctx.actionable_errors = False
    out = run(ctx, "edit_file", path="calc.py", old_text="nope", new_text="x")
    assert out == "ERROR: edit_file failed."


def test_missing_file_suggests_paths(ctx):
    out = run(ctx, "read_file", path="calcs.py")
    assert "does not exist" in out and "Did you mean: calc.py" in out


def test_read_file_numbers_lines_and_pages(ctx):
    out = run(ctx, "read_file", path="calc.py", offset=5, limit=2)
    assert out.splitlines()[0].startswith("calc.py (lines 5-6 of 6)")
    assert "   5| def double(x):" in out


def test_search_returns_path_line_text(ctx):
    out = run(ctx, "search_code", query="def double")
    assert out == "calc.py:5: def double(x):"


def test_search_no_match_is_actionable(ctx):
    assert "Try a shorter or different term" in run(ctx, "search_code", query="zzz")


def test_run_tests_reports_status(ctx):
    assert run(ctx, "run_tests").startswith("FAILED (exit code 1)")
    run(ctx, "edit_file", path="calc.py", old_text="return a - b", new_text="return a + b")
    assert run(ctx, "run_tests").startswith("PASSED")


def test_path_escape_is_denied_even_with_plain_errors(ctx):
    ctx.actionable_errors = False
    out = run(ctx, "read_file", path="../../etc/passwd")
    assert out.startswith("DENIED: path escapes the workspace")


@pytest.mark.parametrize(
    "args,message",
    [
        ({}, "missing required argument(s) ['path']"),
        ({"path": "a", "colour": 1}, "unknown argument(s) ['colour']"),
        ({"path": 3}, "argument 'path' of read_file must be a string"),
        ({"path": "a", "offset": True}, "argument 'offset' of read_file must be a integer"),
    ],
)
def test_schema_validation_messages(args, message):
    assert message in validate_args(TOOLS["read_file"], args)


def test_every_tool_schema_is_strict():
    for tool in TOOLS.values():
        assert tool.parameters["type"] == "object"
        assert tool.parameters["additionalProperties"] is False
        assert tool.description
