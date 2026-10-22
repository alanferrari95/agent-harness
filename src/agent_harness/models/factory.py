"""Build the model for a run from its ID: fake, claude-* or gemini-*."""

from __future__ import annotations

from ..config import ModelConfig
from ..tasks import Task
from .base import Model, ModelError
from .fake import FakeModel


def make_model(model_id: str, task: Task, config: ModelConfig) -> Model:
    """Build the model for one run. The fake model needs the task's policy."""
    if model_id == "fake":
        return FakeModel(task.fake_policy, config.context_window_tokens)
    if model_id.startswith("claude-"):
        from .anthropic_model import AnthropicModel

        return AnthropicModel(model_id, config.max_output_tokens, config.effort)
    if model_id.startswith("gemini-"):
        from .gemini_model import GeminiModel

        return GeminiModel(model_id, config.max_output_tokens)
    raise ModelError(f"unknown model '{model_id}': use fake, claude-* or gemini-*")
