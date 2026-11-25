"""Hard budget enforcement per task (CONSTITUTION §2.10: not advisory)."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from enigma.core.errors import BudgetExceededError
from enigma.core.ids import new_id
from enigma.schemas.common import Actor
from enigma.schemas.governance import Budget, CostRecord
from enigma.store.ledger import CostLedger


class BudgetTracker:
    def __init__(self, budget: Budget, ledger: CostLedger, actor: Actor, task_id: str) -> None:
        self.budget = budget
        self.ledger = ledger
        self.actor = actor
        self.task_id = task_id
        self._started = time.monotonic()
        self.usd = 0.0
        self.tokens = 0
        self.http_requests = 0
        self.cpu_seconds = 0.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def remaining(self) -> Budget:
        return Budget(
            max_usd=max(0.0, self.budget.max_usd - self.usd),
            max_tokens=max(0, self.budget.max_tokens - self.tokens),
            max_http_requests=max(0, self.budget.max_http_requests - self.http_requests),
            max_wall_seconds=max(1e-6, self.budget.max_wall_seconds - self.elapsed),
            max_cpu_seconds=max(1e-6, self.budget.max_cpu_seconds - self.cpu_seconds),
        )

    def check(self) -> None:
        if self.elapsed > self.budget.max_wall_seconds:
            raise BudgetExceededError(
                f"{self.actor} wall time {self.elapsed:.1f}s > {self.budget.max_wall_seconds}s"
            )

    def authorize(
        self, usd: float = 0.0, tokens: int = 0, http_requests: int = 0, cpu_seconds: float = 0.0
    ) -> None:
        """Refuse before spending if the planned spend would exceed any ceiling."""
        self.check()
        checks = (
            ("usd", self.usd + usd, self.budget.max_usd),
            ("tokens", self.tokens + tokens, self.budget.max_tokens),
            ("http_requests", self.http_requests + http_requests, self.budget.max_http_requests),
            ("cpu_seconds", self.cpu_seconds + cpu_seconds, self.budget.max_cpu_seconds),
        )
        for name, planned, ceiling in checks:
            if planned > ceiling and planned > 0:
                raise BudgetExceededError(f"{self.actor} {name} {planned} > ceiling {ceiling}")

    def charge(
        self,
        category: str,
        provider: str,
        usd: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        http_requests: int = 0,
        cpu_seconds: float = 0.0,
        model: str | None = None,
        cached: bool = False,
    ) -> CostRecord:
        """Record actual spend. Spend is recorded even when it breaches, then the breach raises."""
        record = CostRecord(
            id=new_id("cst"),
            actor=self.actor,
            task_id=self.task_id,
            category=category,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=usd,
            http_requests=http_requests,
            cpu_seconds=cpu_seconds,
            cached=cached,
            created_at=datetime.now(UTC),
        )
        self.ledger.record(record)
        self.usd += usd
        self.tokens += input_tokens + output_tokens
        self.http_requests += http_requests
        self.cpu_seconds += cpu_seconds
        if (
            self.usd > self.budget.max_usd
            or self.tokens > self.budget.max_tokens
            or self.http_requests > self.budget.max_http_requests
            or self.cpu_seconds > self.budget.max_cpu_seconds
        ):
            raise BudgetExceededError(f"{self.actor} exceeded budget on task {self.task_id}")
        return record
