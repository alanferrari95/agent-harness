from __future__ import annotations

import pytest

from agent_harness.permissions import PermissionPolicy

RULES = [
    {"tool": "read_file", "decision": "allow"},
    {"tool": "edit_file", "paths": ["tests/*"], "decision": "deny", "reason": "read-only"},
    {"tool": "edit_file", "decision": "ask"},
]


def test_first_matching_rule_wins():
    policy = PermissionPolicy.from_config(RULES)
    assert policy.check("edit_file", {"path": "tests/test_x.py"}).decision == "deny"
    assert policy.check("edit_file", {"path": "./tests/test_x.py"}).decision == "deny"
    assert policy.check("edit_file", {"path": "src/x.py"}).decision == "ask"
    assert policy.check("read_file", {"path": "tests/test_x.py"}).decision == "allow"


def test_undeclared_tools_do_not_exist():
    policy = PermissionPolicy.from_config(RULES)
    assert policy.exposed_tools() == {"read_file", "edit_file"}
    verdict = policy.check("run_tests", {})
    assert verdict.decision == "deny" and verdict.rule_index is None


def test_deny_only_tool_is_not_exposed():
    policy = PermissionPolicy.from_config([{"tool": "run_tests", "decision": "deny"}])
    assert policy.exposed_tools() == set()


def test_invalid_decision_is_rejected():
    with pytest.raises(ValueError):
        PermissionPolicy.from_config([{"tool": "read_file", "decision": "maybe"}])
