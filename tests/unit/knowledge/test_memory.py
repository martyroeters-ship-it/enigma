"""AgentMemory: semantic, episodic, procedural, reflective."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from enigma.core.clock import FrozenClock
from enigma.core.errors import PermissionDeniedError
from enigma.knowledge import AgentMemory
from enigma.schemas.common import AgentRole, Plane
from enigma.schemas.governance import AgentDecision, Reflection, ReflectionOutcome
from enigma.schemas.knowledge import Source, SourceKind
from enigma.store import PlaneView, Stores

NOW = datetime(2026, 11, 25, tzinfo=UTC)


def memory(stores: Stores, role: AgentRole, playbooks: Path | None = None) -> AgentMemory:
    view = PlaneView(
        stores.documents,
        frozenset({Plane.RESEARCH, Plane.CONTROL}),
        frozenset({Plane.RESEARCH, Plane.CONTROL}),
        role.value,
    )
    return AgentMemory(role, view, FrozenClock(NOW), playbooks)


def decision(did: str, agent: AgentRole) -> AgentDecision:
    return AgentDecision(
        id=did,
        agent=agent,
        task_id="task_1",
        decision_type="research.plan",
        outcome="search",
        reason_code="prior_knowledge_empty",
        reasoning_summary="No prior lessons on topic",
        instructions_version="v1",
        created_at=NOW,
    )


def reflection(rid: str, agent: AgentRole) -> Reflection:
    return Reflection(
        id=rid,
        task_id="task_1",
        agent=agent,
        outcome=ReflectionOutcome.SUCCESS,
        what_worked="OpenAlex citation traversal",
        what_failed="Semantic Scholar 429",
        created_at=NOW,
    )


def test_episodic_memory_is_role_scoped(stores: Stores) -> None:
    intel = memory(stores, AgentRole.INTELLIGENCE)
    quant = memory(stores, AgentRole.QUANT_RESEARCH)
    intel.record_decision(decision("dec_1", AgentRole.INTELLIGENCE))
    quant.record_decision(decision("dec_2", AgentRole.QUANT_RESEARCH))
    assert [d.id for d in intel.decisions()] == ["dec_1"]
    with pytest.raises(PermissionDeniedError):
        intel.record_decision(decision("dec_3", AgentRole.QUANT_RESEARCH))


def test_reflective_memory_is_role_scoped(stores: Stores) -> None:
    intel = memory(stores, AgentRole.INTELLIGENCE)
    intel.reflect(reflection("refl_1", AgentRole.INTELLIGENCE))
    assert [r.id for r in intel.reflections()] == ["refl_1"]
    assert memory(stores, AgentRole.CIO).reflections() == []
    with pytest.raises(PermissionDeniedError):
        intel.reflect(reflection("refl_2", AgentRole.CIO))


def test_procedural_playbook_versioned_by_hash(stores: Stores, tmp_path: Path) -> None:
    mem = memory(stores, AgentRole.INTELLIGENCE, tmp_path)
    assert mem.playbook() is None
    (tmp_path / "intelligence.yaml").write_text("search:\n  providers: [arxiv, openalex]\n")
    first = mem.playbook()
    assert first is not None and first.content["search"]["providers"] == ["arxiv", "openalex"]
    (tmp_path / "intelligence.yaml").write_text("search:\n  providers: [openalex]\n")
    second = mem.playbook()
    assert second is not None and second.version_hash != first.version_hash
    assert second.version_hash.startswith("sha256:")
    assert memory(stores, AgentRole.INTELLIGENCE).playbook() is None


def test_semantic_search_uses_knowledge_index(stores: Stores) -> None:
    mem = memory(stores, AgentRole.INTELLIGENCE)
    src = Source(
        id="src_1",
        kind=SourceKind.PREPRINT,
        provider="arxiv",
        url="https://arxiv.org/abs/1",
        title="Perpetual futures funding",
        abstract="Funding rates anchor perpetual prices.",
        content_hash="sha256:" + "b" * 64,
        retrieved_by=AgentRole.INTELLIGENCE,
        retrieved_at=NOW,
    )
    mem.knowledge.add_source(src)
    hits = mem.search("funding perpetual")
    assert [h.id for h in hits] == ["src_1"]
