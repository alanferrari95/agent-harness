"""Versioned harness configuration (configs/v1.yaml, v2.yaml, ...).

Each version changes one thing relative to the previous one and says so in
`changes_from` and `description`, so the eval table can attribute every difference.
Unknown keys are rejected: a typo in a config should fail loudly, not silently fall
back to a default.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .context import CompactionConfig
from .permissions import PermissionPolicy


@dataclass
class LoopConfig:
    max_steps: int = 20
    max_cost_usd: float = 1.0
    max_seconds: float = 300.0
    max_malformed: int = 3  # consecutive responses without a tool call
    stuck_threshold: int = 0  # identical consecutive calls that stop the run; 0 = off
    verify_on_stop: bool = False


@dataclass
class ToolsConfig:
    actionable_errors: bool = False
    test_timeout: int = 120


@dataclass
class ModelConfig:
    id: str = "fake"
    effort: str | None = None  # Anthropic output_config.effort; None = model default
    max_output_tokens: int = 4096
    context_window_tokens: int = 6000  # fake model only: real providers report their own
    num_ctx: int = 16384  # Ollama only: context length sent with every request
    think: bool | None = None  # Ollama only: None = the model's default


@dataclass
class HarnessConfig:
    version: str
    description: str = ""
    changes_from: str | None = None
    loop: LoopConfig = field(default_factory=LoopConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    workspace_mode: str = "copy"
    model: ModelConfig = field(default_factory=ModelConfig)
    permissions: PermissionPolicy = field(default_factory=PermissionPolicy)
    source_path: str | None = None


def _build(cls: type, raw: dict[str, Any] | None, where: str) -> Any:
    raw = raw or {}
    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")
    return cls(**raw)


def load_config(path: str | Path) -> HarnessConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {"version", "description", "changes_from", "loop", "tools", "context",
             "workspace", "model", "permissions"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)}")
    context = raw.get("context") or {}
    workspace = raw.get("workspace") or {}
    return HarnessConfig(
        version=raw["version"],
        description=raw.get("description", "").strip(),
        changes_from=raw.get("changes_from"),
        loop=_build(LoopConfig, raw.get("loop"), f"{path}: loop"),
        tools=_build(ToolsConfig, raw.get("tools"), f"{path}: tools"),
        compaction=_build(CompactionConfig, context.get("compaction"), f"{path}: compaction"),
        workspace_mode=workspace.get("mode", "copy"),
        model=_build(ModelConfig, raw.get("model"), f"{path}: model"),
        permissions=PermissionPolicy.from_config(raw.get("permissions", {}).get("rules", [])),
        source_path=str(path),
    )
