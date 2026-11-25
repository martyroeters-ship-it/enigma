# Enigma — Implementation Plan

**Status:** Ready for implementation team  
**Product:** Enigma  
**Repo (implementation):** `martyroeters-ship-it/enigma`  
**Constraint:** Specs live in Project store `docs/`; this plan does not authorize skipping the Constitution or Validation Protocol.

---

## 0. How to use this document

1. Staff workstreams in §2; respect **dependency order** in §3.
2. Milestone **M0** = scaffolding; **M1** = 13-step autonomous vertical slice (§4) — first institutional proof.
3. Each workstream lists **interface contracts** it must publish before dependents integrate.
4. Do not implement live capital. BTC/ETH spot + perps only for M1.
5. Validation outcomes are programmatic — no LLM pass/fail.

---

## 1. Milestone map (summary)

| Milestone | Goal | Exit gate |
|-----------|------|-----------|
| **M0** | Monorepo skeleton, schemas, FSM stub, CI, three-plane docker networks | Packages import; empty FSM persists state; CI green |
| **M1** | 13-step autonomous research→validation slice (§4) | All §4 acceptance tests pass |
| **M2** | Full validation suite v1 + paper-trading path (single strategy) | Protocol v1 gates + paper loop |
| **M3** | Portfolio + risk veto + multi-strategy | Risk clearance required for paper allocate |
| **M4** | Meta-learning + tech-innovation adoption loops | ImprovementProposal FSM with measured lift |

This document specifies **M0/M1 in depth**. M2–M4 are outlined for sequencing only.

---

## 2. Parallel workstreams

Eleven streams. Owners may be humans or agent sessions; contracts are mandatory either way.

### WS1 — Core / orchestration

| | |
|--|--|
| **Mission** | Shared kernel + explicit institutional FSM + scheduler/outbox |
| **Owns** | `enigma-core`, `apps/scheduler`, task registry, correlation IDs, clock/hash utilities |
| **Does not own** | Agent prompts, market connectors, gate math |
| **Interface contracts** | |
| | `Task` `{id, type, status, assignee_role, payload_ref, budget, checkpoint}` |
| | `FsmEvent` `{aggregate_id, from_state, to_state, reason_code, actor, at, payload_hash}` |
| | `OutboxMessage` envelope per ARCHITECTURE.md §5.2 |
| | API: `create_task`, `checkpoint_task`, `resume_task`, `transition(aggregate, event)` |
| **Depends on** | Nothing (foundation) |
| **Unblocks** | All streams |

### WS2 — Agent infra & communication

| | |
|--|--|
| **Mission** | Role contracts, session runtime, budgets, typed comms |
| **Owns** | `enigma-agents`, `enigma-comm`, role YAML under `registries/agents/` |
| **Interface contracts** | |
| | `RoleContract` (mission, allow/deny tools, memory scopes, KPI refs, budget ceilings) |
| | `AgentSession` start/stop/resume with budget accounting |
| | Message types: `TaskAssigned`, `TaskCompleted`, `TaskFailed`, plus domain messages below |
| | Tool gateway: allowlist enforcement + audit log |
| **Depends on** | WS1 |
| **Unblocks** | WS4, WS6, WS8, WS9 (callable agents) |

### WS3 — Memory / knowledge

| | |
|--|--|
| **Mission** | Institutional registry, claim levels, provenance, agent memories |
| **Owns** | `enigma-knowledge`, Postgres registry schema, object key conventions |
| **Interface contracts** | |
| | CRUD/version APIs for Source, Evidence, Hypothesis, Experiment, StrategyPackage, ValidationReport, Reflection, Lesson |
| | `claim_level` promotion API (L3 write = validation service only) |
| | Memory APIs: semantic search (optional), episodic append, procedural get, reflective append |
| | `LessonRecord` `{id, source_run_id, insight, evidence_refs, created_by}` |
| **Depends on** | WS1, WS10 (authz hooks) |
| **Unblocks** | WS4, WS6, WS8 |

### WS4 — Intelligence engine

| | |
|--|--|
| **Mission** | Real external research: fetch public sources, extract evidence, dedupe, cite |
| **Owns** | Intelligence Agent tools, crawler adapters, source normalization |
| **Interface contracts** | |
| | `ResearchQuery` `{topic, constraints, budget}` |
| | `SourceIngested` `{source_id, url, retrieved_at, content_hash, license_notes}` |
| | `EvidenceExtracted` `{evidence_id, source_id, claims[], citations[], claim_level: L0}` |
| | No L1+ promotion authority |
| **Depends on** | WS2, WS3, WS10 (egress allowlist) |
| **Unblocks** | WS6 (source-backed hypotheses) |

### WS5 — Market data

| | |
|--|--|
| **Mission** | Acquire/normalize/version BTC/ETH spot+perp history; PIT checks |
| **Owns** | `enigma-data`, DatasetVersion publisher, quality reports |
| **Interface contracts** | |
| | `DatasetVersion` schema (KNOWLEDGE_ARCHITECTURE) |
| | `get_bars(dataset_version_id, instrument, start, end)` → framed data |
| | `DataIntegrityAlert` events |
| | Research vs validation dataset catalogs are separate configs |
| **Depends on** | WS1, WS10 |
| **Unblocks** | WS6, WS7 |

### WS6 — Quant research

| | |
|--|--|
| **Mission** | Falsifiable hypotheses + research experiments from evidence + data |
| **Owns** | `enigma-research` (hypothesis/experiment), Quant Research agent wiring |
| **Interface contracts** | |
| | `HypothesisProposed` (mechanism, prediction, falsification, family_id) |
| | `ExperimentRegistered` / results (research-plane metrics only) |
| | Feature/label builders with explicit event-time API |
| **Depends on** | WS2, WS3, WS4, WS5 |
| **Unblocks** | WS7 |

### WS7 — Backtesting & validation

| | |
|--|--|
| **Mission** | Sandboxed strategy execution, reproducible backtests, independent gates |
| **Owns** | Strategy runner sandbox, backtest engine, `enigma-validation`, holdout access |
| **Interface contracts** | |
| | `StrategyPackage` build + `content_hash` |
| | `run_backtest(package_id, dataset_version_id, cost_model_version, seed) → artifact` |
| | `ValidationRequest` / `ValidationReport` (overall computed in code) |
| | Gate library G01+ per VALIDATION_PROTOCOL (`protocol_version: 0.1.0-slice` for M1) |
| **Depends on** | WS1, WS3, WS5, WS6, WS10 |
| **Unblocks** | M1 exit, WS8 (metrics), M2 paper |

### WS8 — Meta-learning

| | |
|--|--|
| **Mission** | Reflections → lessons → improvement proposals; measure lift |
| **Owns** | Meta-Learning agent, ImprovementProposal FSM |
| **Interface contracts** | |
| | `Reflection` write after each M1 run |
| | `LessonRecord` persist |
| | `ImprovementProposal` / `ImprovementDecision` |
| | `NextExperimentProposal` `{parent_run_id, hypothesis_delta, rationale_refs}` |
| **Depends on** | WS2, WS3, WS7 (for run outcomes) |
| **Unblocks** | M4; **partially required in M1** for steps 10–12 |

### WS9 — Tech innovation

| | |
|--|--|
| **Mission** | Track/prototype/benchmark tools; no unilateral adopt |
| **Owns** | Innovation watchlist, spike sandbox templates |
| **Interface contracts** | |
| | `InnovationProposal` `{benchmark_plan, baseline_ref, rollback}` |
| **Depends on** | WS2, WS3 |
| **Unblocks** | M4; **stub only in M1** (contract + no-op registry) |

### WS10 — Security / testing

| | |
|--|--|
| **Mission** | Isolation, secrets, sandboxing, audit, gate unit/integration tests |
| **Owns** | Network policies, secret layout, sandbox profiles, `tests/validation_gates` |
| **Interface contracts** | |
| | Plane identity (`research` \| `validation` \| `execution`) on every worker |
| | Secret providers per plane; deny cross-plane credential read |
| | Kill-switch stub API (halts execution plane) |
| | Test fixtures: leaked-feature package must FAIL G01 |
| **Depends on** | WS1 (hooks) |
| **Unblocks** | Safe WS4/5/7 integration |

### WS11 — Dashboard

| | |
|--|--|
| **Mission** | Human visibility: run timeline, artifacts, reports, resume controls |
| **Owns** | `apps/web` + `enigma-ui-api` read APIs |
| **Interface contracts** | |
| | Read: run status, FSM state, sources, hypothesis, validation report summary, checkpoints |
| | Action: `resume_run(run_id)` → WS1 (authn later) |
| **Depends on** | WS1, WS3, WS7 read models |
| **Unblocks** | Operator UX; **minimal CLI acceptable for M1** if API exists |

---

## 3. Dependency order

```text
WS1 Core/orchestration
  ├─ WS10 Security/testing  (parallel early)
  ├─ WS2 Agent infra & communication
  │     ├─ WS3 Memory/knowledge  (+ WS10 authz)
  │     │     ├─ WS4 Intelligence
  │     │     ├─ WS9 Tech innovation (stub)
  │     │     └─ WS8 Meta-learning (after first outcomes; wire interfaces early)
  │     └─ WS6 Quant research  ← also needs WS4 + WS5
  ├─ WS5 Market data
  └─ WS7 Backtesting & validation  ← WS5 + WS6 + WS3 + WS10
        └─ WS11 Dashboard (after read models stable)
```

**Critical path for M1:** WS1 → (WS10 ∥ WS2 ∥ WS5) → WS3 → WS4 → WS6 → WS7 → WS8 (reflection/lessons/next) → WS11 optional.

---

## 4. Milestone M1 — 13-step autonomous vertical slice

### 4.1 Objective

Demonstrate one **autonomous** Enigma run that: accepts an objective, delegates work, performs **real external research**, stores **source-backed** knowledge, forms a **falsifiable** hypothesis, pulls **historical** BTC/ETH data, executes a strategy in an **isolated** sandbox, produces a **reproducible** backtest, runs **independent programmatic validation**, **reflects**, stores **lessons**, proposes the **next experiment**, and can **resume after interruption**.

No live capital. No LLM validation judgment.

### 4.2 The 13 steps (normative)

| Step | Name | Required outcome | Primary WS |
|------|------|------------------|------------|
| 1 | **Objective** | Typed `RunObjective` persisted (market scope, success signal for the *process*, not “make money”) | WS1, WS11 |
| 2 | **Delegation** | CIO/scheduler emits `TaskAssigned` to roles with budgets and correlation_id | WS1, WS2 |
| 3 | **Real external research** | ≥1 live HTTP fetch of public research/docs (not fixtures-only); `SourceIngested` with URL + content_hash | WS4 |
| 4 | **Source-backed knowledge** | `EvidenceExtracted` nodes link to `source_id`; claim_level L0 | WS3, WS4 |
| 5 | **Falsifiable hypothesis** | `Hypothesis` with mechanism, prediction, falsification criteria, family_id | WS6 |
| 6 | **Historical data** | `DatasetVersion` for BTC and/or ETH perp/spot bars; `pit_certified=true` or explicit waiver logged | WS5 |
| 7 | **Isolated strategy execution** | Strategy code runs in sandbox (no network, resource limits); `StrategyPackage` hash | WS7, WS10 |
| 8 | **Reproducible backtest** | Two runs same seed/package/data → metrics within tolerance; artifact URI stored | WS7 |
| 9 | **Independent validation** | `ValidationReport` with `protocol_version: 0.1.0-slice`; `overall` from code | WS7 |
| 10 | **Reflection** | Structured `Reflection` for the run (success or fail) | WS8 |
| 11 | **Lessons stored** | ≥1 `LessonRecord` linked to run_id and evidence/hypothesis refs | WS8, WS3 |
| 12 | **Proposed next experiment** | `NextExperimentProposal` with rationale referencing lessons/validation | WS8, WS6 |
| 13 | **Resume after interruption** | Kill worker mid-run (after step 3 checkpoint); `resume_task` continues without duplicating side effects | WS1, WS2 |

### 4.3 M1 protocol gates (`0.1.0-slice`)

Mandatory for step 9: **G01, G02, G03 (smoke), G06 (ledger present), G07 (compute even if threshold soft), G12, G16**.  
Document soft vs hard thresholds in `registries/validation/protocol_0.1.0-slice.yaml`. Full suite = M2.

### 4.4 Acceptance tests (M1)

Automated unless marked manual.

| ID | Test | Pass criteria |
|----|------|---------------|
| AT1 | Objective intake | POST/CLI creates run; state `OBJECTIVE_SET` |
| AT2 | Delegation | ≥2 roles receive `TaskAssigned` with same `correlation_id` |
| AT3 | Live research | Test env allowlists one real URL; source row shows non-fixture `retrieved_at` and hash |
| AT4 | Provenance graph | Evidence → Source FK enforced; orphan evidence rejected |
| AT5 | Hypothesis quality | Registration API rejects missing mechanism or falsification |
| AT6 | Dataset publish | DatasetVersion content_hash stable; bars query returns rows in range |
| AT7 | Sandbox isolation | Strategy attempting outbound socket fails; audit log records deny |
| AT8 | Reproducibility | Twin backtests: max abs metric delta ≤ configured ε |
| AT9 | Validation determinism | Twin validation reports identical `overall` + gate results |
| AT10 | Leakage fail | Fixtured leaky package → G01 FAIL, `overall=FAIL` |
| AT11 | No LLM gate | Mutation test: LLM adapter returning “PASS” cannot flip `overall` |
| AT12 | Reflection+lesson | Rows exist, FK to run_id |
| AT13 | Next experiment | Proposal references parent_run_id + lesson_id |
| AT14 | Resume | Interrupt after research checkpoint; resume completes steps 4–12 once; outbox idempotent |
| AT15 | Plane secrets | Research worker cannot read validation holdout credentials (integration) |
| AT16 | Manual demo | Operator shows dashboard or CLI timeline of all 13 steps (manual) |

### 4.5 Explicit non-goals for M1

- Live trading / real capital  
- Full 12-agent LLM concurrency  
- Portfolio optimization  
- Neo4j / heavy graph  
- Next.js polish beyond read timeline  
- PyTorch/RL search  
- Multi-exchange routing  

---

## 5. Milestone M0 — acceptance (pre-slice)

- Repo `enigma` packages importable  
- Postgres migrations: tasks, outbox, fsm_events  
- Docker compose: `research`, `validation`, `execution` networks  
- CI: lint + unit tests on core/schemas  
- Cursor rules copied from store drafts into `.cursor/rules/`  

---

## 6. Milestones M2–M4 (sketch)

| M2 | Enable remaining gates; paper broker; degradation alerts; human risk acknowledge |
| M3 | Portfolio agent; correlation constraints; risk veto path wired to execution |
| M4 | Meta-learning adoption with measured lift; tech innovation benchmark admit/reject |

Each requires its own acceptance list before start (ADR).

---

## 7. Cross-cutting SLAs for implementers

- Every public function at plane boundaries: typed (Pydantic)  
- Every transition: audit event  
- Every experiment: registered **before** run  
- Cost model version always set on backtest  
- No fabricated “operational” demos — tests must hit real code paths  

---

## 8. Open uncertainties (blockers to clarify early)

1. Workspace tool: `uv` vs Poetry — pick in M0 ADR.  
2. First backtest engine: VectorBT vs custom — ADR-003; must not block sandbox interface.  
3. Holdout: separate Postgres/MinIO instance vs schema — prefer separate instance.  
4. Soft thresholds in `0.1.0-slice` vs failing the slice when edge is weak: **process can PASS institutionally even if strategy FAIL** — run success ≠ alpha success. Encode this in AT9/AT12.  
5. Dashboard mandatory vs CLI for AT16 — CLI sufficient if timeline is complete.
