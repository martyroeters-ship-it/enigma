# Enigma — instructions for coding agents

Enigma's *runtime* agent contracts (the twelve institutional roles) are in
[docs/AGENTS.md](docs/AGENTS.md). This file is for development agents working on the code.

## Read first
- [docs/CONSTITUTION.md](docs/CONSTITUTION.md) — binding. `.cursor/rules/` enforces it.
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) — M1 13-step slice and AT1–AT16.
- [docs/adr/](docs/adr/) — implementation ADRs (ADR-011+) that refine `docs/ARCHITECTURAL_DECISIONS.md`.

## Commands
```bash
uv sync                       # install (Python 3.12)
uv run pytest -m "not network"  # offline tests (CI gate)
uv run pytest -m network      # real public-API tests
uv run ruff check . && uv run ruff format --check .
uv run mypy                   # strict
uv run lint-imports           # plane isolation contracts
```

## Conventions
- All cross-module data are Pydantic models in `enigma.schemas` (frozen, `extra="forbid"`).
  Add new contracts there; register persisted kinds with `enigma.store.register_kind`.
- Persistence only via `enigma.store` (`DocumentStore`/`PlaneView`, `TaskQueue`, `AuditLog`,
  `CostLedger`, `Checkpoints`). No raw SQL elsewhere.
- IDs: `enigma.core.new_id("<prefix>")`. Hashes: `enigma.core.sha256_of(obj)`.
- Never read wall-clock time in strategy or validation logic; inject `enigma.core.Clock`.
- Tests that touch real external services are marked `@pytest.mark.network`. Offline tests
  may use recorded responses, but recorded data must never be presented as live operation.
- No mocks presented as operational. If a provider is not configured, raise
  `ProviderUnavailableError`; do not fabricate output.
