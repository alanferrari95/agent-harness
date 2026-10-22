from __future__ import annotations

import json

import pytest
from conftest import REPO_ROOT

from agent_harness.config import load_config
from agent_harness.evals import (
    assert_no_regression,
    comparison_table,
    inject,
    load_summaries,
    run_suite,
)
from agent_harness.tasks import load_task, load_tasks

CONFIGS = sorted((REPO_ROOT / "configs").glob("v*.yaml"), key=lambda p: int(p.stem[1:]))


# ------------------------------------------------------------------ gate (as in invoice-mlops)


def test_gate_passes_at_baseline():
    assert assert_no_regression({"success_rate": 0.9}, {"success_rate": 0.9}) == (
        "success_rate=0.9 >= baseline=0.9"
    )


def test_gate_fails_below_baseline_and_names_the_tasks():
    baseline = {"success_rate": 0.9, "tasks": {"a": {"successes": 1}, "b": {"successes": 1}}}
    current = {"success_rate": 0.8, "tasks": {"a": {"successes": 1}, "b": {"successes": 0}}}
    with pytest.raises(RuntimeError, match=r"below baseline.*regressed tasks: \['b'\]"):
        assert_no_regression(current, baseline)


def test_gate_fails_if_success_rate_missing():
    with pytest.raises(RuntimeError, match="missing key"):
        assert_no_regression({"mean_steps": 3}, {"success_rate": 0.5})


# ------------------------------------------------------------------ configs


def test_every_config_loads():
    versions = [load_config(path).version for path in CONFIGS]
    assert versions == [f"v{i}" for i in range(1, len(CONFIGS) + 1)]


def _knobs(path) -> dict:
    cfg = load_config(path)
    return {
        "loop": vars(cfg.loop),
        "tools": vars(cfg.tools),
        "compaction": vars(cfg.compaction),
        "workspace": cfg.workspace_mode,
        "model": vars(cfg.model),
        "permissions": [vars(rule) for rule in cfg.permissions.rules],
    }


def _flatten(prefix: str, value) -> dict:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_flatten(f"{prefix}.{key}", item))
        return out
    return {prefix: value}


@pytest.mark.parametrize("previous,current", list(zip(CONFIGS, CONFIGS[1:], strict=False)))
def test_each_version_changes_one_thing(previous, current):
    before, after = _flatten("", _knobs(previous)), _flatten("", _knobs(current))
    changed = [key for key in before.keys() | after.keys() if before.get(key) != after.get(key)]
    # Compaction is one feature with three settings; everything else is one key.
    groups = {key.split(".")[1] if ".compaction." in key else key for key in changed}
    assert len(groups) == 1, f"{current.name} changes {sorted(changed)}"
    assert load_config(current).changes_from == load_config(previous).version


# ------------------------------------------------------------------ tasks and the suite


def test_every_task_fixture_starts_red():
    tasks = load_tasks(REPO_ROOT / "evals" / "tasks")
    assert 8 <= len(tasks) <= 12
    for task in tasks:
        assert task.goal and task.fake_policy.get("steps"), task.id
        assert (task.repo / "tests").is_dir(), task.id


def test_suite_writes_summary_and_report(tmp_path):
    config = load_config(CONFIGS[-1])
    task = load_task(REPO_ROOT / "evals" / "tasks" / "like_button")
    summary = run_suite(config, "fake", [task], out_dir=tmp_path, log=lambda *_: None)
    assert summary["success_rate"] == 1.0 and summary["tokens_estimated"]
    saved = json.loads((tmp_path / config.version / "fake" / "summary.json").read_text())
    assert saved["tasks"]["like_button"]["successes"] == 1
    table = comparison_table(load_summaries(tmp_path))
    assert f"| {config.version} |" in table and "1/1 (100%)" in table


def test_inject_replaces_only_between_markers():
    doc = "intro\n<!-- results:start -->\nold\n<!-- results:end -->\noutro"
    assert inject(doc, "new") == "intro\n<!-- results:start -->\nnew\n<!-- results:end -->\noutro"
