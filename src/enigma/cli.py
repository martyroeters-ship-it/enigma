"""Enigma command-line interface."""

from __future__ import annotations

import json

import typer

from enigma.core.config import Settings
from enigma.fsm import ALL_MACHINES, StateMachineStore
from enigma.store import Stores

app = typer.Typer(
    help="Enigma — autonomous quantitative research institution", no_args_is_help=True
)


def _stores() -> tuple[Settings, Stores]:
    settings = Settings.from_env()
    settings.ensure_dirs()
    return settings, Stores.open(settings.db_path)


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
    }
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
