"""Thin per-role agent memory: semantic, episodic, procedural, reflective (WS3 memory APIs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from enigma.core.clock import Clock
from enigma.core.errors import IntegrityError, PermissionDeniedError
from enigma.core.hashing import sha256_hex
from enigma.knowledge.base import INDEXED_KINDS, KnowledgeBase
from enigma.knowledge.models import KnowledgeHit
from enigma.schemas.common import AgentRole, StrictModel
from enigma.schemas.governance import AgentDecision, Reflection
from enigma.store import PlaneView


class Playbook(StrictModel):
    role: AgentRole
    version_hash: str
    path: str
    content: dict[str, Any]


class AgentMemory:
    def __init__(
        self,
        role: AgentRole,
        view: PlaneView,
        clock: Clock,
        playbooks_dir: Path | None = None,
    ) -> None:
        self.role = role
        self.view = view
        self.clock = clock
        self.playbooks_dir = playbooks_dir
        self.knowledge = KnowledgeBase(view, clock, actor=role)

    # semantic
    def search(
        self, query: str, kinds: tuple[str, ...] = INDEXED_KINDS, limit: int = 10
    ) -> list[KnowledgeHit]:
        """Full-text search scoped to the planes this role's view may read."""
        return self.knowledge.retrieve(query, kinds=kinds, limit=limit)

    # episodic
    def record_decision(self, decision: AgentDecision) -> str:
        if decision.agent != self.role:
            raise PermissionDeniedError(
                f"{self.role} may not record decisions for {decision.agent}"
            )
        self.view.put(decision)
        return decision.id

    def decisions(self, limit: int = 50) -> list[AgentDecision]:
        return self.view.list(AgentDecision, where={"agent": self.role.value}, limit=limit)

    # procedural
    def playbook(self) -> Playbook | None:
        """Versioned playbook from `<playbooks_dir>/<role>.yaml`; None if absent."""
        if self.playbooks_dir is None:
            return None
        path = self.playbooks_dir / f"{self.role.value}.yaml"
        if not path.is_file():
            return None
        raw = path.read_bytes()
        content = yaml.safe_load(raw) or {}
        if not isinstance(content, dict):
            raise IntegrityError(f"playbook {path} must be a YAML mapping")
        return Playbook(
            role=self.role,
            version_hash="sha256:" + sha256_hex(raw),
            path=path.as_posix(),
            content=content,
        )

    # reflective
    def reflect(self, reflection: Reflection) -> str:
        if reflection.agent != self.role:
            raise PermissionDeniedError(
                f"{self.role} may not reflect on behalf of {reflection.agent}"
            )
        self.view.put(reflection)
        return reflection.id

    def reflections(self, limit: int = 50) -> list[Reflection]:
        return self.view.list(Reflection, where={"agent": self.role.value}, limit=limit)
