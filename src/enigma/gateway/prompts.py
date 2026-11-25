"""Versioned prompt templates (`registries/prompts/*.yaml`).

Every LLM-backed agent step renders a registered template. The template's `version` and the
content hash of its text are recorded with the step's decision, so a change in methodology is
always a new, identifiable prompt version rather than a silent edit.

File format::

    id: quant.hypothesis
    version: "1"
    methodology: "Short statement of the method this prompt encodes."
    system: |
      ...
    user: |
      ... {placeholders} ...
"""

from __future__ import annotations

import string
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from enigma.core.hashing import sha256_of


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    version: str
    methodology: str
    system: str
    user: str

    @property
    def content_hash(self) -> str:
        return sha256_of(
            {"id": self.id, "version": self.version, "system": self.system, "user": self.user}
        )

    @property
    def ref(self) -> str:
        """Stable reference recorded with decisions: ``id@version#hash12``."""
        return f"{self.id}@{self.version}#{self.content_hash.removeprefix('sha256:')[:12]}"

    def placeholders(self) -> set[str]:
        return {
            name
            for text in (self.system, self.user)
            for _, name, _, _ in string.Formatter().parse(text)
            if name
        }

    def render(self, **values: Any) -> tuple[str, str]:
        """Return (system, user). Missing or unexpected placeholders raise KeyError."""
        expected = self.placeholders()
        missing = expected - values.keys()
        extra = values.keys() - expected
        if missing or extra:
            raise KeyError(
                f"prompt {self.id}: missing {sorted(missing)}, unexpected {sorted(extra)}"
            )
        return self.system.format(**values), self.user.format(**values)


class PromptRegistry:
    def __init__(self, templates: Mapping[str, PromptTemplate]) -> None:
        self._templates = dict(templates)

    @classmethod
    def load(cls, directory: Path) -> PromptRegistry:
        templates: dict[str, PromptTemplate] = {}
        for path in sorted(directory.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            template = PromptTemplate(
                id=str(raw["id"]),
                version=str(raw["version"]),
                methodology=str(raw.get("methodology", "")),
                system=str(raw["system"]),
                user=str(raw["user"]),
            )
            if template.id in templates:
                raise ValueError(f"duplicate prompt id {template.id} in {path}")
            templates[template.id] = template
        return cls(templates)

    def get(self, prompt_id: str) -> PromptTemplate:
        try:
            return self._templates[prompt_id]
        except KeyError:
            raise KeyError(f"no registered prompt {prompt_id!r}") from None

    def ids(self) -> list[str]:
        return sorted(self._templates)
