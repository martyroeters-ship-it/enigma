# Enigma

Autonomous quantitative research institution: discovers hypotheses from real external research,
tests them on real historical BTC/ETH data in an isolated sandbox, validates them with
deterministic, programmatic gates, and learns from every outcome, including failures.

**No live trading.** No real capital is ever allocated by this codebase in the current phase
(CONSTITUTION §2.9, ADR-010).

## Status
Milestone M1 (13-step autonomous vertical slice) in progress. See
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) and the milestone reports.

## Quick start
```bash
uv sync
uv run enigma init
uv run enigma status
```

## Layout
```text
src/enigma/
  core/        ids, clocks, hashing, errors, config, logging
  schemas/     typed Pydantic contracts (knowledge, research, validation, governance, messages)
  store/       SQLite control DB: versioned documents, audit chain, task queue, ledgers
  fsm/         explicit state machines (strategy lifecycle, research cycle, proposals)
registries/    human-owned versioned registries (validation protocol, agent contracts)
docs/          institutional specs + implementation ADRs (docs/adr/)
.cursor/rules/ Cursor rules enforcing the constitution
```
Further subpackages (agents, gateway, intelligence, knowledge, data, research, backtest,
validation, learning, orchestrator, api) land per workstream; see docs/adr/ADR-011.
