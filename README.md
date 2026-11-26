# Enigma

Autonomous quantitative research institution: discovers hypotheses from real external research,
tests them on real historical BTC/ETH data in an isolated sandbox, validates them with
deterministic, programmatic gates, and learns from every outcome, including failures.

**No live trading.** No real capital is ever allocated by this codebase in the current phase
(CONSTITUTION §2.9, ADR-010). Paper trading is also disabled in M1 (`registries/risk/limits.yaml`).

## Status
Milestone M1 (13-step autonomous vertical slice). See
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md), [docs/adr/](docs/adr/) and the
milestone reports. No credentials are required; see [docs/CREDENTIALS.md](docs/CREDENTIALS.md).

## Quick start
```bash
uv sync
uv run enigma init
uv run enigma research          # submit the default BTC/ETH funding objective and run it
uv run enigma dashboard         # read-only dashboard on http://127.0.0.1:8000
```

Operating commands:

| Command | What it does |
|---|---|
| `enigma research "<objective>" [--symbols BTCUSDT,ETHUSDT] [--no-run]` | Step 1: submit an objective; by default run until the cycle finishes |
| `enigma run [--until-idle]` | Autonomous loop: drives cycles, executes agent tasks, chains next experiments (daily cap), runs scheduled technology scans; stops cleanly on SIGINT/SIGTERM |
| `enigma resume` | Requeue tasks orphaned by a dead worker and finish every interrupted cycle |
| `enigma cycle <id>` | JSON timeline of one cycle: FSM events, tasks, hypothesis, experiments, validation, lessons, next proposal |
| `enigma status` | Queue, state machines, costs, audit-chain verification |
| `enigma kill-switch on\|off` | Halt or allow all agent work |
| `enigma credentials` | Which optional credentials are configured (never prints values) |

State lives under `ENIGMA_HOME` (default `./.enigma`): `control/` (SQLite system of record),
`research/` (research-plane data, raw sources, backtest artifacts), `validation/` (holdout data
and reports; only the validation service opens it).

## The 13-step cycle
`OBJECTIVE_RECEIVED → DELEGATED → RESEARCH_RETRIEVED → KNOWLEDGE_STORED → HYPOTHESIS_GENERATED →
DATA_ACQUIRED → STRATEGY_EXECUTED → BACKTEST_RECORDED → VALIDATION_DECIDED → REFLECTED →
LESSONS_STORED → NEXT_EXPERIMENT_PROPOSED → COMPLETED` (step 13 is resumability: every step is
checkpointed and every delegated task has a deterministic id, so `enigma resume` continues
exactly where a crash stopped). The CIO (`orchestrator/cycle.py`) is deterministic policy; the
other roles (`orchestrator/roles.py`) reach capabilities only through allowlisted, audited tools
(`orchestrator/bindings.py`).

## Layout
```text
src/enigma/
  core/          ids, clocks, hashing, errors, config, logging
  schemas/       typed Pydantic contracts
  store/         SQLite control DB: versioned documents, audit chain, task queue, ledgers, FTS
  fsm/           explicit state machines (strategy lifecycle, research cycle, proposals)
  comm/          outbox message bus
  agents/        role contracts, budgets, tool gateway, delegation, worker loop
  gateway/       provider-independent model gateway (fails closed without keys)
  intelligence/  recursive external research (arXiv, OpenAlex, Crossref, GitHub, Wikipedia, ...)
  knowledge/     provenance-checked knowledge base and agent memory
  data/          Binance public archives + OKX funding, PIT checks, versioned datasets
  research/      mechanism library, hypothesis generation, microstructure statistics
  strategy/      strategy interface, static checks, network-namespace sandbox
  backtest/      deterministic bar engine and statistics
  validation/    holdout custody, deterministic gates, ValidationService (no model imports)
  risk/          deterministic risk service
  learning/      lessons, next-experiment proposals, institutional review
  orchestrator/  CIO driver, role agents, tool bindings, composition root, daemon
  api/           read-only dashboard and JSON API
registries/      human-owned registries: agents, validation protocol, costs, risk, models, schedules
docs/            institutional specs + implementation ADRs (docs/adr/)
.cursor/rules/   Cursor rules enforcing the constitution
```
