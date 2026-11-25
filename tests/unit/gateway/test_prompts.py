from __future__ import annotations

from pathlib import Path

import pytest

from enigma.gateway.prompts import PromptRegistry

TEMPLATE = """id: demo.step
version: "1"
methodology: Demo.
system: |
  You are {role}.
user: |
  Question: {question}
"""


def test_render_and_version_reference(tmp_path: Path) -> None:
    (tmp_path / "demo.yaml").write_text(TEMPLATE)
    prompt = PromptRegistry.load(tmp_path).get("demo.step")
    system, user = prompt.render(role="an analyst", question="why?")
    assert system.strip() == "You are an analyst."
    assert "why?" in user
    assert prompt.ref.startswith("demo.step@1#")
    with pytest.raises(KeyError):
        prompt.render(role="x")
    with pytest.raises(KeyError):
        prompt.render(role="x", question="y", other="z")


def test_content_change_changes_hash(tmp_path: Path) -> None:
    (tmp_path / "demo.yaml").write_text(TEMPLATE)
    before = PromptRegistry.load(tmp_path).get("demo.step").content_hash
    (tmp_path / "demo.yaml").write_text(TEMPLATE.replace("Question", "Query"))
    assert PromptRegistry.load(tmp_path).get("demo.step").content_hash != before


def test_repository_prompts_load() -> None:
    root = Path(__file__).resolve().parents[3] / "registries" / "prompts"
    registry = PromptRegistry.load(root)
    for prompt_id in registry.ids():
        assert registry.get(prompt_id).version
