# Enigma — System Architecture

**Status:** Draft for review  
**Product:** Enigma  
**GitHub repository:** `martyroeters-ship-it/enigma`  
**Related:** [CONSTITUTION.md](./CONSTITUTION.md) · [AGENTS.md](./AGENTS.md) · [SECURITY.md](./SECURITY.md) · [TECHNOLOGY_STACK.md](./TECHNOLOGY_STACK.md) · [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)

---

## 1. Architectural Intent

Enigma is an institutional quantitative research platform: discover hypotheses, implement experiments, adversarially validate them with programmatic gates, and (later) allocate paper/live capital under risk veto.

**Design doctrine:**

- Prefer **simple explicit state machines** over complex orchestration frameworks.
- Prefer **Python core** for quant, agents, validation, and data pipelines.
- Enforce **research / validation / execution isolation**.
- Treat agents as **logical roles** (not necessarily twelve separate models).
- No live capital in the initial phase; BTC/ETH spot + perps first.

---

## 2. Proposed Repository Structure

Greenfield monorepo `enigma`:

```text
enigma/
├── README.md
├── AGENTS.md                      # symlink or copy of agent contracts summary
├── CONSTITUTION.md                # ratified copy
├── pyproject.toml                 # workspace root (uv/poetry TBD — open Q)
├── docker/
│   ├── research.Dockerfile
│   ├── validation.Dockerfile
│   └── execution.Dockerfile
├── .cursor/rules/                 # copied from docs/cursor-rules/
├── docs/                          # institutional specs (source of truth in store until repo exists)
├── packages/
│   ├── enigma-core/               # shared types, IDs, errors, clocks, hashing
│   ├── enigma-schemas/            # Pydantic models + JSON Schema export
│   ├── enigma-data/               # ingestion, PIT integrity, Parquet/DuckDB access
│   ├── enigma-research/           # hypothesis, feature, experiment runners
│   ├── enigma-validation/         # gates, metrics, holdout client (no LLM judge)
│   ├── enigma-portfolio/          # allocation, correlation, regime (post-slice)
│   ├── enigma-risk/               # limits, veto API, kill switch
│   ├── enigma-execution/          # paper broker adapters (CCXT later)
│   ├── enigma-knowledge/          # registry graph + provenance APIs
│   ├── enigma-agents/             # role contracts, budgets, memory adapters
│   ├── enigma-comm/               # message bus / protocol adapters
│   └── enigma-ui-api/             # FastAPI control plane (read-heavy initially)
├── apps/
│   ├── worker-research/
│   ├── worker-validation/
│   ├── worker-execution/          # dormant for live; paper only when enabled
│   ├── scheduler/                 # state-machine driver (not LangGraph-first)
│   └── web/                       # Next.js console (deferred until vertical slice works)
├── experiments/                   # versioned experiment configs + manifests
├── registries/                    # committed methodology versions (YAML)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── validation_gates/          # golden tests for statistical gates
└── scripts/
    ├── bootstrap_dev.sh
    └── run_vertical_slice.sh
```

**Challenge to directive sprawl:** Do not staff all twelve packages on day one. **M1 critical packages:** `enigma-core`, `enigma-schemas`, `enigma-data`, `enigma-research`, `enigma-validation`, `enigma-knowledge`, `enigma-agents`, `enigma-comm`, plus `apps/scheduler` and plane workers. Defer `enigma-portfolio`, `enigma-risk` (stub veto OK), `enigma-execution`, `enigma-ui-api`/web until after M1 gates. See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md).

---

## 3. Logical System Context

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Control Plane (CIO)                       │
│              explicit state machine + task registry              │
└────────────┬──────────────────┬──────────────────┬──────────────┘
             │                  │                  │
     ┌───────▼──────┐   ┌───────▼──────┐   ┌───────▼──────┐
     │   RESEARCH   │   │  VALIDATION  │   │  EXECUTION   │
     │    plane     │──▶│    plane     │──▶│    plane     │
     │  (discover)  │   │  (invalidate)│   │ (paper/live) │
     └───────┬──────┘   └───────┬──────┘   └───────┬──────┘
             │                  │                  │
             ▼                  ▼                  ▼
        research DB         holdout vault      exec vault
        + object store      (isolated)         (isolated)
             │                  │                  │
             └──────────┬───────┴──────────┬───────┘
                        ▼                  ▼
                 Knowledge Registry    Risk Veto Service
                 (shared read model)   (independent authority)
```

Data flows **downstream** as versioned artifacts (hypothesis → strategy package → validation report → deployment ticket). Upstream feedback is **events and proposals**, never raw holdout leakage into research.

---

## 4. State Models

### 4.1 Run FSM (M1 institutional process)

Autonomous slice run (process success ≠ alpha success):

```text
OBJECTIVE_SET
  → DELEGATED
  → RESEARCHING_EXTERNAL
  → KNOWLEDGE_RECORDED
  → HYPOTHESIS_REGISTERED
  → DATA_READY
  → STRATEGY_RUNNING
  → BACKTEST_RECORDED
  → VALIDATION_RUNNING
  → VALIDATION_RECORDED      # overall PASS or FAIL both allowed
  → REFLECTED
  → LESSONS_STORED
  → NEXT_EXPERIMENT_PROPOSED
  → COMPLETED
```

Checkpoints after each step enable **resume** (step 13). Side effects idempotent on `correlation_id` + step key.

### 4.2 Strategy artifact FSM (promotion path)

```text
DRAFT_HYPOTHESIS
  → RESEARCH_EXPERIMENT_RUNNING
  → RESEARCH_CANDIDATE
  → VALIDATION_QUEUED
  → VALIDATION_RUNNING
  → VALIDATION_FAILED | VALIDATION_PASSED
  → PAPER_APPROVAL_PENDING   # M2+; Risk non-veto + human gate
  → PAPER_RUNNING
  → PAPER_DEGRADED | PAPER_STOPPED | LIVE_CANDIDATE (future)
  → ARCHIVED
```

**Rules:**

- Transitions are append-only events with actor, reason code, artifact hashes.
- `VALIDATION_FAILED` cannot transition to paper/live without a **new** package ID.
- Risk Management may force halt from any post-validation execution state (when execution exists).
- Improvement proposals use a separate FSM (`PROPOSED → BENCHMARKING → …`) per [LEARNING_ARCHITECTURE.md](./LEARNING_ARCHITECTURE.md).

Open uncertainty: portfolio construction as parallel machine vs gate on `PAPER_APPROVAL_PENDING` — see [ARCHITECTURE_REVIEW.md](./ARCHITECTURE_REVIEW.md).

---

## 5. Communication Protocol

### 5.1 Principles

- Structured messages only at agent boundaries (Pydantic / JSON Schema).
- No free-form “please deploy” without typed fields and artifact IDs.
- At-least-once delivery with idempotent handlers.
- Correlation IDs bind task → experiment → validation → decision.

### 5.2 Message envelope

```json
{
  "message_id": "uuid",
  "correlation_id": "uuid",
  "timestamp_utc": "ISO-8601",
  "from_agent": "quantitative_research",
  "to_agent": "independent_validation",
  "message_type": "ValidationRequest",
  "schema_version": "1.0.0",
  "payload_hash": "sha256:...",
  "payload": {}
}
```

### 5.3 Transport (phased)

| Phase | Transport | Rationale |
|-------|-----------|-----------|
| M0–M1 | PostgreSQL outbox + worker poll | Simple, auditable, no broker ops |
| M2+ | Optional NATS/Redis streams **if** latency/backpressure demands | Adopt only with benchmark evidence |
| Reject initially | Complex multi-agent orchestration frameworks | Unnecessary vs explicit FSM |

### 5.4 Core message types

- `TaskAssigned`, `TaskCompleted`, `TaskFailed`
- `HypothesisProposed`, `ExperimentRegistered`
- `ValidationRequest`, `ValidationReport`
- `RiskVeto`, `RiskClearance`
- `ImprovementProposal`, `ImprovementDecision`
- `KillSwitchActivated`

Full schemas: [KNOWLEDGE_ARCHITECTURE.md](./KNOWLEDGE_ARCHITECTURE.md) § Data Schemas and `enigma-schemas` package (future).

---

## 6. Dependency Analysis

### 6.1 Hard dependencies (M1 slice)

```text
enigma-core + apps/scheduler (Run FSM + outbox)
  └─ enigma-schemas
       ├─ enigma-comm + enigma-agents
       ├─ enigma-knowledge
       ├─ enigma-data ─────────────────────────────┐
       ├─ intelligence tools (WS4) → knowledge     │
       └─ enigma-research ← data + knowledge       │
              └─ enigma-validation ← packages + holdout
                     └─ meta-learning writes (reflection/lesson/next)
```

Full workstream dependency graph: [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) §3.

### 6.2 Soft / deferred dependencies

- Portfolio, execution (paper), UI — after first validated candidate path exists.
- Knowledge graph richness — start with relational registry; add graph/vector retrieval when retrieval quality is measured.
- Heavy ML (PyTorch) — defer until non-ML baselines and validation harness exist.

### 6.3 Isolation dependency rule

Validation **must not** import research feature pipelines that can recompute labels with future data. Shared code is limited to `enigma-core` + pure metric libraries reviewed for leakage.

---

## 7. Runtime Topology (Initial)

- Single Docker Compose stack for local/dev with three network namespaces (research / validation / execution).
- PostgreSQL (control + registry), DuckDB/Parquet (analytics), object store (MinIO or local FS) for artifacts.
- One scheduler process driving FSMs; agents invoked as role-scoped tool sessions with budgets.

Production topology is **out of scope** until paper-trading milestone; do not invent Kubernetes requirements prematurely.

---

## 8. Minimal End-to-End Vertical Slice

**Normative definition:** the **13-step autonomous slice** in [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) §4 (objective → delegation → real external research → source-backed knowledge → falsifiable hypothesis → historical data → isolated strategy execution → reproducible backtest → independent validation → reflection → lessons → next experiment → resume after interruption).

**Acceptance tests:** IMPLEMENTATION_PLAN §4.4 (AT1–AT16).  
**Protocol:** `0.1.0-slice` per [VALIDATION_PROTOCOL.md](./VALIDATION_PROTOCOL.md).  
**Process vs alpha:** run may COMPLETE with `ValidationReport.overall=FAIL` (ADR-007).

**Explicitly out of slice:** live trading, portfolio, full 12-agent LLM concurrency, RL/evolutionary search, Next.js polish, dual backtest engines.

---

## 9. Challenges to the Directive (Architecture)

| Directive tendency | Challenge | Evidence / trade-off |
|--------------------|-----------|----------------------|
| 12 specialized agents from day one | Start with **role contracts + shared runtime**; instantiate LLM sessions on demand | Reduces cost/complexity; preserves specialization via contracts |
| Rich knowledge graph immediately | Relational registry first; graph/vector when needed | Premature graph = schema churn without retrieval metrics |
| Broad methodology catalogue | Vertical slice picks **one** hypothesis class | Prevents validation harness from being blocked by research breadth |
| Sophisticated orchestration | Explicit FSM + outbox | Debuggable, auditable; frameworks add opacity |

---

## 10. Open Architectural Uncertainties

Documented for human review — **not silently resolved**:

1. Package/workspace tool: `uv` vs Poetry vs Hatch.
2. Whether CIO is a deterministic policy engine with optional LLM advise, or an LLM with tool constraints (recommend: **deterministic FSM + LLM advisor**).
3. Holdout custody: separate DB instance vs separate schema + IAM (recommend separate instance for true isolation).
4. When paper trading begins relative to portfolio agent readiness.
5. Single-exchange vs multi-exchange for slice data (recommend single to reduce microstructure confounders).
6. Whether VectorBT or a thin custom event backtester owns the first engine (see ADR).

See [ARCHITECTURAL_DECISIONS.md](./ARCHITECTURAL_DECISIONS.md) and [ARCHITECTURE_REVIEW.md](./ARCHITECTURE_REVIEW.md).
