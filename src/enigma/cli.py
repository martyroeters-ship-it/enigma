"""Enigma command-line interface."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Annotated

import httpx
import typer
import yaml

from enigma.core.config import Settings
from enigma.fsm import ALL_MACHINES, RESEARCH_CYCLE, StateMachineStore
from enigma.store import Stores
from enigma.store.ledger import KILL_SWITCH

if TYPE_CHECKING:
    from enigma.orchestrator.daemon import Daemon
    from enigma.orchestrator.institution import Institution

app = typer.Typer(
    help="Enigma — autonomous quantitative research institution", no_args_is_help=True
)

DEFAULT_OBJECTIVE = (
    "Do perpetual-futures funding rates contain information about subsequent BTC and ETH "
    "returns that survives realistic costs and out-of-sample validation?"
)


def _stores() -> tuple[Settings, Stores]:
    settings = Settings.from_env()
    settings.ensure_dirs()
    return settings, Stores.open(settings.db_path)


def _institution() -> Institution:
    from enigma.orchestrator.institution import Institution  # noqa: PLC0415 - heavy imports
    from enigma.orchestrator.services import default_services  # noqa: PLC0415

    return Institution.build(Settings.from_env(), default_services)


def _daemon(inst: Institution) -> Daemon:
    from enigma.orchestrator.daemon import Daemon, Schedule  # noqa: PLC0415

    return Daemon(inst, Schedule.load(inst.settings.registries_dir / "schedules.yaml"))


def _require_spend_and_trading(settings: Settings) -> None:
    if settings.max_daily_api_spend_usd != 0:
        typer.echo("ENIGMA_MAX_DAILY_API_SPEND_USD must be 0. Refusing to start.")
        raise typer.Exit(code=1)
    limits_path = settings.registries_dir / "risk" / "limits.yaml"
    limits = yaml.safe_load(limits_path.read_text(encoding="utf-8")) or {}
    paper_off = limits.get("paper_trading_enabled") is False
    live_off = limits.get("live_trading_enabled") is False
    if not paper_off or not live_off:
        typer.echo(f"{limits_path} must keep paper and live trading disabled. Refusing to start.")
        raise typer.Exit(code=1)


def _require_ollama() -> str:
    try:
        response = httpx.get("http://127.0.0.1:11434/api/version", timeout=3.0)
        response.raise_for_status()
    except Exception as exc:
        typer.echo(f"Ollama is not reachable at http://127.0.0.1:11434 ({exc}).")
        raise typer.Exit(code=1) from exc
    version = response.json().get("version", "unknown")
    return str(version)


def _require_strategy_isolation() -> str:
    from enigma.strategy.sandbox import SandboxRunner, SandboxUnavailableError  # noqa: PLC0415

    try:
        mode = SandboxRunner(allow_weak=False).check_isolation("startup")
    except SandboxUnavailableError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    return mode


@app.command()
def init() -> None:
    """Create state directories and migrate the control database."""
    settings, stores = _stores()
    typer.echo(f"Enigma home: {settings.home}")
    typer.echo(f"Control DB schema version: {stores.db.schema_version}")


@app.command()
def status() -> None:
    """Print queue, state-machine, cost, and audit-chain status as JSON."""
    _, stores = _stores()
    fsm = StateMachineStore(stores.db, stores.audit)
    report = {
        "tasks": stores.queue.counts(),
        "machines": {m.name: fsm.state_counts(m) for m in ALL_MACHINES},
        "costs": stores.costs.totals(),
        "audit_entries_verified": stores.audit.verify(),
        "kill_switch": stores.flags.kill_switch_active(),
        "cycles": [
            {"id": c["workflow_id"], "step": c["step"], "status": c["status"]}
            for c in stores.checkpoints.list("research_cycle")
        ],
    }
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command()
def research(
    objective: Annotated[str, typer.Argument()] = DEFAULT_OBJECTIVE,
    symbols: str = "BTCUSDT,ETHUSDT",
    topics: str = "funding rate,perpetual futures,crowding,basis",
    max_trials: int = 9,
    run: bool = typer.Option(True, help="Run the institution until the cycle finishes"),
) -> None:
    """Submit a research objective (step 1) and, by default, run the cycle to completion."""
    from enigma.orchestrator.institution import perp  # noqa: PLC0415

    settings = Settings.from_env()
    _require_spend_and_trading(settings)
    _require_ollama()
    mode = _require_strategy_isolation()
    os.environ.setdefault("ENIGMA_HYPOTHESIS_MODE", "llm")
    typer.echo(f"strategy sandbox: {mode}")
    inst = _institution()
    already = set(inst.driver.active_cycles())
    try:
        cycle_id = inst.driver.submit_objective(
            objective,
            tuple(perp(s.strip().upper()) for s in symbols.split(",") if s.strip()),
            tuple(t.strip() for t in topics.split(",") if t.strip()),
            max_trials=max_trials,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    if cycle_id in already:
        typer.echo(f"cycle {cycle_id} already running for this objective")
    else:
        typer.echo(f"cycle {cycle_id} submitted")
    if run:
        _daemon(inst).run(until_idle=True)
        typer.echo(f"cycle {cycle_id}: {inst.fsm.state(RESEARCH_CYCLE, cycle_id)}")


@app.command()
def up() -> None:
    """Check spend, trading flags, Ollama, and the strategy sandbox, then run."""
    settings = Settings.from_env()
    _require_spend_and_trading(settings)
    version = _require_ollama()
    mode = _require_strategy_isolation()
    os.environ.setdefault("ENIGMA_HYPOTHESIS_MODE", "llm")
    inst = _institution()
    daemon = _daemon(inst)
    daemon.install_signal_handlers()
    typer.echo(f"home {settings.home}")
    typer.echo(f"Ollama {version}; spend cap $0; paper and live trading off")
    typer.echo(f"strategy sandbox: {mode}")
    typer.echo(f"hypothesis mode {os.environ.get('ENIGMA_HYPOTHESIS_MODE')}")
    schedule = daemon.schedule
    if schedule.continuous and schedule.max_cycles_per_day <= 0:
        typer.echo(
            "continuous research: on; no daily cycle cap; "
            f"{schedule.max_concurrent_cycles} cycle at a time; "
            "the backlog opens the next objective"
        )
    else:
        typer.echo(f"continuous research: off; daily cycle cap {schedule.max_cycles_per_day}")

    def _progress(iteration: int, status: str) -> None:
        typer.echo(f"step {iteration}: {status}")

    try:
        n = daemon.run(until_idle=False, on_step=_progress)
    except RuntimeError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(f"daemon stopped after {n} iterations")


@app.command(name="run")
def run_daemon(
    until_idle: bool = typer.Option(False, help="Exit when no cycle is active"),
    max_iterations: int | None = None,
) -> None:
    """Run the autonomous loop (cycles, tasks, experiment chaining, schedules)."""
    _require_strategy_isolation()
    inst = _institution()
    daemon = _daemon(inst)
    daemon.install_signal_handlers()
    n = daemon.run(until_idle=until_idle, max_iterations=max_iterations)
    typer.echo(f"daemon stopped after {n} iterations")


@app.command()
def resume() -> None:
    """Requeue orphaned tasks and continue every interrupted cycle until idle."""
    _require_strategy_isolation()
    inst = _institution()
    daemon = _daemon(inst)
    daemon.install_signal_handlers()
    active = inst.driver.active_cycles()
    typer.echo(f"resuming {len(active)} cycle(s): {', '.join(active) or '-'}")
    daemon.run(until_idle=True)
    for cycle_id in active:
        typer.echo(f"cycle {cycle_id}: {inst.fsm.state(RESEARCH_CYCLE, cycle_id)}")


@app.command()
def cycle(cycle_id: str) -> None:
    """Print one cycle's timeline, tasks and outcome as JSON."""
    from enigma.api.readmodels import ReadModels  # noqa: PLC0415

    _, stores = _stores()
    detail = ReadModels(stores).cycle_detail(cycle_id)
    detail.pop("messages")
    typer.echo(json.dumps(detail, indent=2, default=str))


@app.command()
def dashboard(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the read-only operational dashboard."""
    import uvicorn  # noqa: PLC0415

    from enigma.api.app import create_app  # noqa: PLC0415

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


@app.command(name="kill-switch")
def kill_switch(state: Annotated[str, typer.Argument(help="on | off")]) -> None:
    """Halt (on) or allow (off) all agent work."""
    if state not in ("on", "off"):
        raise typer.BadParameter("state must be 'on' or 'off'")
    _, stores = _stores()
    stores.flags.set(KILL_SWITCH, state, "human")
    stores.audit.record("human", "kill_switch.set", None, {"state": state})
    typer.echo(f"kill switch {state}")


sandbox_app = typer.Typer(help="Strategy sandbox backends")
app.add_typer(sandbox_app, name="sandbox")


@sandbox_app.command("status")
def sandbox_status() -> None:
    """Print whether strategy code can run in a network namespace or a Linux container."""
    from enigma.strategy.sandbox import SandboxRunner, SandboxUnavailableError  # noqa: PLC0415

    try:
        mode = SandboxRunner().check_isolation("status")
    except SandboxUnavailableError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(mode)


@sandbox_app.command("prepare")
def sandbox_prepare() -> None:
    """Build the local Linux sandbox image and probe that its network stays closed."""
    from enigma.strategy.sandbox_docker import prepare_sandbox_image  # noqa: PLC0415

    try:
        detail = prepare_sandbox_image()
    except RuntimeError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(detail)


@app.command()
def credentials() -> None:
    """List optional credentials and which are configured (values are never printed)."""
    from enigma.orchestrator.services import credential_report  # noqa: PLC0415

    typer.echo(json.dumps(credential_report(), indent=2))


if __name__ == "__main__":
    app()
