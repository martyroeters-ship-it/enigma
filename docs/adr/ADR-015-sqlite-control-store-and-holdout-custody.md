# ADR-015 — SQLite control store for M1; separate validation store for holdout custody

**Status:** Accepted for M1 (implementation team, 2026-11-25) · **Refines:** ADR-002, ADR-005, ADR-006

## Context
The specs assume PostgreSQL (outbox, registry) and Docker Compose with three networks. The M1
build environment has neither Docker nor PostgreSQL, and the directive forbids paid
infrastructure. The system must still run continuously, resume after interruption, and keep
holdout data out of reach of the research plane.

## Decision
- **Control/registry store: SQLite (WAL)** at `$ENIGMA_HOME/control/enigma.sqlite3`: versioned
  documents, FSM events, outbox, leased task queue, cost ledger, checkpoints, hash-chained audit
  log, FTS5 search. All access goes through `enigma.store` repositories (no SQL in callers).
- **Holdout custody:** holdout series and holdout manifests live only under
  `$ENIGMA_VALIDATION_DIR` in a **separate SQLite database and directory**, opened only by the
  validation service. The research plane receives only `RedactedValidationSummary`.
  `PlaneView` blocks research-plane reads of `ValidationReport` at the repository layer.
- Docker Compose with per-plane networks is deferred until a host with Docker is available;
  per-process network isolation for strategy code is provided by ADR-016 instead.

## Consequences
- Zero-infrastructure, single-file backups; runs on any host with Python 3.12.
- PostgreSQL migration path: the schema is portable (no SQLite-only features other than FTS5,
  replaceable with `tsvector`).

## Alternatives
- Embedded PostgreSQL binaries: adds ops burden without a measured need at M1 volume.

## Limitations
- Single-writer concurrency; adequate for one scheduler and a handful of workers.
- Holdout isolation is **filesystem- and process-level on one host, not separate credentials or
  machines**. That is weaker than the separate instance preferred by ADR-006; AT15 tests cover
  the repository and process boundary, not a hostile local user.
