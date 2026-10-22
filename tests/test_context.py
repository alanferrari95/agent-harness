from __future__ import annotations

from agent_harness.context import CompactionConfig, ContextBuilder
from agent_harness.state import RunState, StepRecord


def records(n: int, size: int = 400) -> list[StepRecord]:
    out = []
    for i in range(1, n + 1):
        tool = "read_file" if i % 2 else "run_tests"
        obs = (f"file {i}\n" + "x" * size) if tool == "read_file" else f"FAILED (exit code 1)\n{i}"
        out.append(StepRecord(i, tool, {"path": f"f{i}.py"} if tool == "read_file" else {}, obs))
    return out


def builder(enabled: bool, threshold: int = 200, keep: int = 2) -> ContextBuilder:
    return ContextBuilder("Do the task.", "a.py\nb.py", "Be careful.", CompactionConfig(enabled, threshold, keep))


def test_stable_sections_come_first():
    prompt = builder(False).build(RunState(), records(2))
    order = ["# Task", "# Project files", "# Repository instructions", "# Harness state",
             "# Recent steps", "# Last observation"]
    positions = [prompt.index(title) for title in order]
    assert positions == sorted(positions)


def test_without_compaction_everything_is_verbatim():
    ctx = builder(False)
    history = records(8)
    assert not ctx.maybe_compact(history)
    prompt = ctx.build(RunState(), history)
    assert all(f"- step {i}:" in prompt for i in range(1, 8))
    assert "# Earlier steps" not in prompt


def test_compaction_folds_whole_steps_and_keeps_the_tail():
    ctx = builder(True)
    history = records(8)
    assert ctx.maybe_compact(history)
    assert ctx.folded == 6
    prompt = ctx.build(RunState(), history)
    assert "(6 earlier steps folded into this summary)" in prompt
    assert "Files read: f1.py; f3.py; f5.py" in prompt
    assert "- step 6:" not in prompt and "- step 7:" in prompt
    assert "# Last observation\nstep 8:" in prompt


def test_summary_is_merged_incrementally():
    ctx = builder(True)
    history = records(8)
    ctx.maybe_compact(history)
    history += records(14)[8:]
    ctx.maybe_compact(history)
    rendered = ctx.summary.render()
    assert ctx.compactions == 2
    assert rendered.startswith("(12 earlier steps folded")
    assert "f1.py" in rendered and "f11.py" in rendered  # early facts survive


def test_compaction_respects_threshold():
    ctx = builder(True, threshold=100_000)
    assert not ctx.maybe_compact(records(8))
