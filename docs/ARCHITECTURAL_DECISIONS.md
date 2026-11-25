# Enigma — Architectural Decisions (ADR log)

**Status:** Living draft  
**Product:** Enigma  
**Rule:** Every significant decision records context, decision, consequences, alternatives, limitations. Open items stay open until human-ratified.

---

## ADR-001 — Monorepo and Python package layout

| | |
|--|--|
| **Status** | Proposed |
| **Context** | Greenfield `martyroeters-ship-it/enigma`; multiple planes and libraries |
| **Decision** | Single monorepo with `packages/enigma-*` and `apps/*`; Python packages installable via workspace tool |
| **Consequences** | Atomic cross-package PRs; clearer plane boundaries via apps |
| **Alternatives** | Polyrepo per plane — slower for M1 |
| **Limitations** | Requires discipline to avoid circular imports |
| **Open** | uv vs Poetry vs Hatch — **recommend uv**, not ratified |

---

## ADR-002 — Orchestration = explicit FSM + Postgres outbox

| | |
|--|--|
| **Status** | Proposed (strong recommendation) |
| **Context** | Need auditable, resumable workflows; directive prefers simple state machines |
| **Decision** | CIO/scheduler drives typed FSM; commands via outbox table; workers poll/claim idempotently |
| **Consequences** | Resume-after-interrupt (M1 step 13) is natural; ops are debuggable |
| **Alternatives** | LangGraph/Temporal/Celery — defer until FSM proven insufficient |
| **Limitations** | Not optimal for ultra-low-latency trading triggers (out of M1 scope) |

---

## ADR-003 — Backtest engine behind interface

| | |
|--|--|
| **Status** | Open |
| **Context** | VectorBT is fast; NautilusTrader is realistic; custom is controllable |
| **Decision** | **Pending.** Implement `BacktestEngine` protocol first; choose VectorBT **or** thin custom for M1 funding/bars strategy — not both |
| **Evidence needed** | Spike: funding-rate strategy correctness vs speed on BTC perp bars |
| **Limitation** | Wrong choice may force rewrite of cost hooks |

---

## ADR-004 — Agents as role contracts on shared runtime

| | |
|--|--|
| **Status** | Proposed |
| **Context** | Twelve logical agents; separate model deployments are costly and opaque |
| **Decision** | One runtime; twelve RoleContracts; LLM sessions on demand; validation/risk core paths deterministic |
| **Consequences** | Matches AGENTS.md; reduces M1 surface |
| **Alternatives** | 12 microservices — rejected for M1 |

---

## ADR-005 — Knowledge: relational registry first

| | |
|--|--|
| **Status** | Proposed |
| **Context** | Rich graph + vectors tempting but unmeasured |
| **Decision** | PostgreSQL entities/edges tables for M1; pgvector deferred; no Neo4j |
| **Consequences** | Provenance queries via SQL; semantic search later |
| **Alternatives** | Graph DB day one — rejected as over-engineering |

---

## ADR-006 — Holdout custody

| | |
|--|--|
| **Status** | Proposed |
| **Context** | Isolation mandatory |
| **Decision** | Prefer **separate validation object store + credentials**; control metadata may share Postgres with strict no-holdout-series rule |
| **Open** | Fully separate Postgres instance vs schema — security prefers instance; cost may push schema for solo-dev M1 with compensating tests |

---

## ADR-007 — M1 success ≠ alpha success

| | |
|--|--|
| **Status** | Proposed |
| **Context** | Slice must complete even if strategy fails validation |
| **Decision** | Run FSM can reach `COMPLETED` with `ValidationReport.overall=FAIL`; lessons still stored; next experiment proposed |
| **Consequences** | Acceptance tests distinguish institutional process pass vs edge pass |
| **Alternatives** | Require PASS to finish slice — rejected (incentivizes soft gates) |

---

## ADR-008 — Market scope lock for M1

| | |
|--|--|
| **Status** | Proposed |
| **Context** | Directive: BTC/ETH spot + perps |
| **Decision** | Single venue historical OHLCV + funding for BTC/ETH perps sufficient for M1; spot optional if basis hypothesis chosen |
| **Open** | Which venue — pick one with clear historical API ToS |

---

## ADR-009 — Dashboard optional for M1

| | |
|--|--|
| **Status** | Proposed |
| **Context** | WS11 exists but critical path is process |
| **Decision** | CLI/API timeline satisfies AT16 if complete; Next.js deferred without blocking M1 |
| **Consequences** | WS11 can trail WS7 |

---

## ADR-010 — No live capital until explicit later milestone

| | |
|--|--|
| **Status** | Accepted (constitutional) |
| **Decision** | Execution plane paper-only when introduced; live keys absent from repo and compose |
| **Consequences** | Simplifies security review for M1–M2 |

---

## Rejected / parked ideas

| Idea | Why parked |
|------|------------|
| Always-on 12 LLM agents | Cost, nondeterminism; use contracts + on-demand |
| Dual backtest engines in M1 | Interface one engine |
| Crawl4AI+Playwright required for step 3 | httpx allowlisted fetch enough |
| Kubernetes for M1 | Compose sufficient |
