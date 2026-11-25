"""Versioned role contracts (docs/AGENTS.md). Loaded from `registries/agents/*.yaml`.

Contracts are human-owned files. Agents read them; they cannot modify them. The canonical hash of a
contract is its `instructions_version` and is stamped on every decision the agent records.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field

from enigma.core.hashing import sha256_of
from enigma.schemas.common import AgentRole, NonEmptyStr, Plane, StrictModel
from enigma.schemas.governance import Budget


class KpiDefinition(StrictModel):
    id: NonEmptyStr
    definition: NonEmptyStr
    direction: NonEmptyStr = Field(description="higher_is_better | lower_is_better | target_zero")


class RoleContract(StrictModel):
    role: AgentRole
    number: int = Field(ge=1, le=12)
    name: NonEmptyStr
    version: NonEmptyStr
    mission: NonEmptyStr
    plane: Plane
    readable_planes: tuple[Plane, ...]
    writable_planes: tuple[Plane, ...]
    task_kinds: tuple[str, ...] = Field(description="Task kinds this role may execute")
    may_delegate_to: tuple[AgentRole, ...] = ()
    tools: tuple[str, ...] = Field(description="Tool allowlist (least privilege)")
    authority: tuple[str, ...]
    denied: tuple[str, ...]
    memory_scopes: tuple[str, ...]
    kpis: tuple[KpiDefinition, ...]
    budget: Budget
    model_route: str | None = Field(default=None, description="Gateway routing key, if any")
    deterministic_core: bool = Field(
        default=False, description="Decisions are made by deterministic code (validation, risk)"
    )
    instructions: NonEmptyStr

    @property
    def instructions_version(self) -> str:
        return sha256_of(self)


class ContractRegistry:
    def __init__(self, contracts: dict[AgentRole, RoleContract]) -> None:
        missing = set(AgentRole) - set(contracts)
        if missing:
            raise ValueError(f"missing role contracts: {sorted(m.value for m in missing)}")
        self._contracts = contracts

    @classmethod
    def load(cls, directory: Path) -> ContractRegistry:
        contracts: dict[AgentRole, RoleContract] = {}
        for path in sorted(directory.glob("*.yaml")):
            contract = RoleContract.model_validate(yaml.safe_load(path.read_text("utf-8")))
            if path.stem != contract.role.value:
                raise ValueError(f"{path.name} must be named {contract.role.value}.yaml")
            contracts[contract.role] = contract
        return cls(contracts)

    def get(self, role: AgentRole) -> RoleContract:
        return self._contracts[role]

    def all(self) -> list[RoleContract]:
        return sorted(self._contracts.values(), key=lambda c: c.number)

    def role_for_task(self, kind: str) -> AgentRole:
        owners = [c.role for c in self._contracts.values() if kind in c.task_kinds]
        if len(owners) != 1:
            raise ValueError(f"task kind {kind!r} must have exactly one owner, found {owners}")
        return owners[0]

    @property
    def version(self) -> str:
        return sha256_of({r.value: c.instructions_version for r, c in self._contracts.items()})
