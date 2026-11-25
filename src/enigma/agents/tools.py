"""Tool gateway: allowlist enforcement + audit for every agent tool call."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from enigma.core.errors import PermissionDeniedError
from enigma.core.hashing import sha256_of
from enigma.schemas.common import Plane
from enigma.store.audit import AuditLog


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    plane: Plane
    fn: Callable[..., Any]
    # When true the gateway passes the calling role as `actor`, so tools can never be invoked
    # on behalf of another role.
    pass_actor: bool = False


class ToolRegistry:
    """Composition-root registry of capabilities. Agents never get this object directly."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)


class ToolGateway:
    """Per-session view of tools: only the contract allowlist, only the agent's planes."""

    def __init__(
        self,
        registry: ToolRegistry,
        allowlist: tuple[str, ...],
        planes: frozenset[Plane],
        actor: str,
        audit: AuditLog,
        task_id: str,
    ) -> None:
        self._registry = registry
        self._allow = frozenset(allowlist)
        self._planes = planes
        self.actor = actor
        self._audit = audit
        self._task_id = task_id

    def available(self) -> list[str]:
        return sorted(n for n in self._allow if self._registry.get(n) is not None)

    def call(self, name: str, /, **kwargs: Any) -> Any:
        tool = self._registry.get(name)
        denied_reason = None
        if name not in self._allow:
            denied_reason = "not_in_allowlist"
        elif tool is None:
            denied_reason = "not_registered"
        elif tool.plane not in self._planes:
            denied_reason = f"plane_{tool.plane.value}_not_permitted"
        if denied_reason is not None or tool is None:
            self._audit.record(
                self.actor,
                "tool.denied",
                self._task_id,
                {"tool": name, "reason": denied_reason},
            )
            raise PermissionDeniedError(f"{self.actor} may not call tool {name}: {denied_reason}")
        self._audit.record(
            self.actor,
            "tool.call",
            self._task_id,
            {"tool": name, "args_hash": sha256_of(_hashable(kwargs))},
        )
        if tool.pass_actor:
            if "actor" in kwargs:
                raise PermissionDeniedError(f"{self.actor} may not supply actor to {name}")
            return tool.fn(actor=self.actor, **kwargs)
        return tool.fn(**kwargs)


def _hashable(kwargs: dict[str, Any]) -> dict[str, str]:
    return {k: repr(v)[:500] for k, v in sorted(kwargs.items())}
