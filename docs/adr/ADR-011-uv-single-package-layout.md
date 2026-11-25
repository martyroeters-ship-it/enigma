# ADR-011 — uv + single `enigma` package with plane subpackages (resolves ADR-001)

**Status:** Accepted (implementation team, 2026-11-25) · **Supersedes:** open item in ADR-001

## Context
ADR-001 proposed `packages/enigma-*` plus `apps/*` and left uv vs Poetry vs Hatch open. The M1 team
is small and parallel workstreams need atomic cross-cutting changes.

## Decision
- **uv** for environments and locking (`uv.lock` committed); hatchling build backend.
- One distributable package `enigma` under `src/enigma/`, with subpackages per concern:
  `core`, `schemas`, `store`, `fsm`, `agents`, `comm`, `gateway`, `intelligence`, `knowledge`,
  `data`, `research`, `strategy`, `backtest`, `validation`, `risk`, `execution`, `learning`,
  `orchestrator`, `api`.
- Plane boundaries are **enforced in CI by import-linter contracts** (`pyproject.toml`), e.g.
  research/intelligence cannot import validation or execution; validation cannot import research.

## Consequences
- One lockfile, one CI job, one version. Boundaries are machine-checked instead of implied by
  directory names.
- Process-level isolation (separate workers per plane, separate state directories and
  databases) is still required and is handled by ADR-015 and ADR-016; import contracts only stop
  accidental code coupling.

## Alternatives
- uv workspace with 12+ packages: more ceremony, same isolation guarantee unless deployed
  separately. Can be split later along the existing subpackage boundaries.
- Poetry: slower resolver, no advantage here.

## Limitations
- A single wheel means a deployed worker technically ships all code; runtime plane checks
  (`PlaneView`, separate validation DB/dir) are the real guard.
