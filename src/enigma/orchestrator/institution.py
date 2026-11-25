"""Composition root: wires stores, contracts, tools, agents, the CIO driver and the worker."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from enigma.agents import AgentWorker, ContractRegistry
from enigma.comm import MessageBus
from enigma.core.clock import Clock, SystemClock
from enigma.core.config import Settings
from enigma.fsm import StateMachineStore
from enigma.orchestrator.bindings import CoreTools, Services, build_registry
from enigma.orchestrator.cycle import CycleDriver, HoldoutPolicy
from enigma.orchestrator.roles import ALL_ROLE_AGENTS
from enigma.orchestrator.tech import TechnologyInnovationAgent
from enigma.risk.service import RiskLimits, RiskService
from enigma.schemas.common import Instrument
from enigma.store import Stores

ServicesFactory = Callable[[Settings, Stores, HoldoutPolicy, Clock], Services]


def perp(symbol: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        base=symbol.removesuffix("USDT"),
        quote="USDT",
        kind="perpetual",
        venue="binance_usdm",
    )


@dataclass
class Institution:
    settings: Settings
    stores: Stores
    bus: MessageBus
    fsm: StateMachineStore
    contracts: ContractRegistry
    policy: HoldoutPolicy
    driver: CycleDriver
    worker: AgentWorker
    services: Services
    clock: Clock

    @classmethod
    def build(
        cls,
        settings: Settings,
        services_factory: ServicesFactory,
        clock: Clock | None = None,
        lease: timedelta = timedelta(minutes=30),
    ) -> Institution:
        settings.ensure_dirs()
        clock = clock or SystemClock()
        stores = Stores.open(settings.db_path)
        bus = MessageBus(stores.db, stores.audit)
        fsm = StateMachineStore(stores.db, stores.audit)
        reg_dir = settings.registries_dir
        contracts = ContractRegistry.load(reg_dir / "agents")
        policy = HoldoutPolicy.load(reg_dir / "validation" / "holdout_policy.yaml")
        risk = RiskService(stores, RiskLimits.load(reg_dir / "risk" / "limits.yaml"), clock)
        services = services_factory(settings, stores, policy, clock)
        tools = build_registry(CoreTools(stores, fsm, risk, clock), services)
        agents = {a.role: a() for a in (*ALL_ROLE_AGENTS, TechnologyInnovationAgent)}
        worker = AgentWorker(stores, bus, contracts, tools, agents, clock=clock, lease=lease)
        driver = CycleDriver(stores, bus, fsm, contracts, policy, clock)
        return cls(settings, stores, bus, fsm, contracts, policy, driver, worker, services, clock)

    def close(self) -> None:
        self.stores.db.close()
