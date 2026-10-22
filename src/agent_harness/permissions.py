"""Permission rules as data: allow / ask / deny.

Rules live in the version config (YAML), not in the prompt. Evaluation is first match
wins; a rule matches when the tool name is equal and, if the rule lists `paths`, the
call's `path` argument matches one of the globs (fnmatch on the workspace-relative POSIX
path).

What is not declared does not exist: a tool with no allow/ask rule is never shown to
the model, and a call that matches no rule is denied. Two layers sit below any config
and cannot be switched off: the workspace path guard and the `finish` control tool.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Literal

Decision = Literal["allow", "ask", "deny"]
DECISIONS = ("allow", "ask", "deny")

Approver = Callable[[str, dict[str, Any]], bool]


@dataclass(frozen=True)
class Rule:
    tool: str
    decision: Decision
    paths: tuple[str, ...] = ()
    reason: str = ""

    def matches(self, tool: str, args: dict[str, Any]) -> bool:
        if tool != self.tool:
            return False
        if not self.paths:
            return True
        path = str(args.get("path", "")).replace("\\", "/").removeprefix("./")
        return any(fnmatch(path, pattern) for pattern in self.paths)


@dataclass
class Verdict:
    decision: Decision
    rule_index: int | None
    reason: str = ""


@dataclass
class PermissionPolicy:
    rules: list[Rule] = field(default_factory=list)

    @classmethod
    def from_config(cls, raw: list[dict[str, Any]]) -> PermissionPolicy:
        rules = []
        for index, item in enumerate(raw):
            decision = item.get("decision")
            if decision not in DECISIONS:
                raise ValueError(f"permission rule {index}: decision must be one of {DECISIONS}")
            rules.append(
                Rule(
                    tool=item["tool"],
                    decision=decision,
                    paths=tuple(item.get("paths", ())),
                    reason=item.get("reason", ""),
                )
            )
        return cls(rules)

    def exposed_tools(self) -> set[str]:
        """Tools the model may see: those with at least one allow or ask rule."""
        return {rule.tool for rule in self.rules if rule.decision in ("allow", "ask")}

    def check(self, tool: str, args: dict[str, Any]) -> Verdict:
        for index, rule in enumerate(self.rules):
            if rule.matches(tool, args):
                return Verdict(rule.decision, index, rule.reason)
        return Verdict("deny", None, "not declared in the permission rules")


def approve_all(tool: str, args: dict[str, Any]) -> bool:
    """Non-interactive approver for evals and CI. Every approval is still traced."""
    return True


def approve_on_console(tool: str, args: dict[str, Any]) -> bool:
    preview = {k: (v if len(str(v)) < 200 else str(v)[:200] + "...") for k, v in args.items()}
    answer = input(f"Allow {tool}({preview})? [y/N] ")
    return answer.strip().lower() in ("y", "yes")
