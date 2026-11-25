# Enigma — Agent Contracts

**Status:** Draft for review  
**Product:** Enigma  
**Principle:** Agents are logical roles with missions, authority bounds, tools, memory, KPIs, and communication contracts. They need not map 1:1 to separate AI models. Deterministic software owns math, validation gates, risk enforcement, and execution controls.

---

## 1. Universal Agent Capabilities

Every agent contract includes:

| Capability | Requirement |
|------------|-------------|
| Mission | One-sentence institutional purpose |
| Authority | Explicit allow / deny lists |
| Tools | Named tool surface; least privilege |
| Memory | Semantic, episodic, procedural, reflective (versioned) |
| Knowledge access | Scoped reads from institutional registry |
| Communication | Typed messages only (`enigma-comm` schemas) |
| KPIs | Measurable; cannot self-rewrite evaluation criteria |
| Logs | Decision + execution append-only logs |
| Reflection | Post-task structured reflection record |
| Improvement proposals | Via Meta-Learning workflow; never unilateral |
| Versioned instructions | Methodology/prompt SHA required at runtime |
| Budgets | Compute, token, wall-clock, API cost ceilings |

**Hard rule:** Agents may not unilaterally modify evaluation criteria, risk limits, security permissions, or validation requirements ([CONSTITUTION.md](./CONSTITUTION.md)).

---

## 2. Shared Runtime Model

```text
RoleContract (YAML, versioned)
  → AgentSession (budget, memory handles, tool allowlist)
    → Tool calls / Deterministic services
      → Outbox messages + audit events
```

Recommended implementation posture:

- **One orchestration runtime** with twelve role contracts.
- LLM calls only where judgment/search/synthesis is needed.
- Validation Agent and Risk Agent core paths are **deterministic services** with optional LLM commentary that cannot change gate outcomes.

---

## 3. Agent Roster & Contracts

### Agent 01 — Chief Investment Officer (CIO)

| Field | Contract |
|-------|----------|
| **Mission** | Set research priorities, allocate tasks/resources, coordinate workflows, oversee institutional performance. |
| **Authority** | Create/prioritize tasks; advance FSM where policy allows; request validation; convene risk review. |
| **Denied** | Override validation failure; bypass risk veto; allocate real capital; edit constitution/risk limits. |
| **Tools** | Task registry, FSM driver, budget dashboard, agent directory, read-only performance views. |
| **Memory** | Priority agendas; past allocation decisions; institutional KPI snapshots. |
| **KPIs** | Throughput of validated candidates; cycle time research→validation; budget adherence; zero constitutional breaches. |
| **Comms** | Emits `TaskAssigned`; consumes completion/failure and `ValidationReport` summaries. |

**Challenge:** CIO should be a **deterministic policy + scheduler** with optional LLM advisor. An LLM-as-orchestrator increases nondeterminism and audit opacity without proven benefit.

---

### Agent 02 — Meta-Learning Agent

| Field | Contract |
|-------|----------|
| **Mission** | Recursive institutional improvement: evaluate agents, analyze failures, propose methodological upgrades, measure post-adoption effect. |
| **Authority** | Propose versioned changes to prompts, tools, workflows, software; open benchmark tasks. |
| **Denied** | Auto-adopt changes to risk, security, validation criteria; rewrite its own success metrics. |
| **Tools** | Performance analytics, experiment registry queries, A/B harness, version control APIs (PR-only). |
| **Memory** | Improvement proposals; before/after metrics; rejected proposals with reasons. |
| **KPIs** | % proposals with measured lift; false-positive adoption rate; time-to-detect regression. |
| **Comms** | `ImprovementProposal` → Independent Validation / human gate → `ImprovementDecision`. |

---

### Agent 03 — Intelligence Agent

| Field | Contract |
|-------|----------|
| **Mission** | Deep discovery across public sources; citation traversal; evidence extraction; deduplication. |
| **Authority** | Fetch public web/docs; write **unverified** knowledge nodes with provenance. |
| **Denied** | Elevate knowledge claim levels; access holdouts; scrape behind auth without policy approval. |
| **Tools** | Crawl/fetch (Playwright/Crawl4AI evaluated), PDF/HTML extractors, citation graph helpers, dedupe. |
| **Memory** | Source reliability priors; search strategies that worked; dead ends. |
| **KPIs** | Unique high-signal sources/week; provenance completeness; duplicate rate; hallucination incident rate (human/sample audit). |
| **Comms** | `SourceIngested`, `EvidenceExtracted` to Knowledge + Quant Research. |

---

### Agent 04 — Data Engineering Agent

| Field | Contract |
|-------|----------|
| **Mission** | Acquire, normalize, verify, version historical and real-time datasets; detect integrity failures. |
| **Authority** | Publish dataset versions to research store; quarantine bad data; open integrity incidents. |
| **Denied** | Touch validation holdout vault; mutate historical versions in place; mark data “validated edge.” |
| **Tools** | Exchange/data connectors, Polars/Parquet pipelines, PIT validators, quality reports. |
| **Memory** | Vendor quirks; known gaps; successful remediation playbooks. |
| **KPIs** | Freshness SLA; schema break detection time; % datasets with complete PIT metadata; leakage findings caught pre-research. |
| **Comms** | `DatasetVersionPublished`, `DataIntegrityAlert`. |

---

### Agent 05 — Market Microstructure Agent

| Field | Contract |
|-------|----------|
| **Mission** | Model price formation, order flow, liquidity, funding, liquidations, fragmentation, execution behaviour. |
| **Authority** | Publish microstructure analyses and feature specs; advise execution assumptions. |
| **Denied** | Approve strategies; relax slippage models without validation re-run. |
| **Tools** | Order book/trade analytics, funding/OI loaders, simulation of queue/impact (as available). |
| **Memory** | Regime-specific liquidity notes; exchange incident history. |
| **KPIs** | Calibration error of cost models vs paper fills; actionable structure insights cited in passed validations. |
| **Comms** | Advisories consumed by Quant Research, Strategy Engineering, Execution. |

---

### Agent 06 — Quantitative Research Agent

| Field | Contract |
|-------|----------|
| **Mission** | Original alpha discovery via math/stats/ML with structural/causal explanations — not indicator soup. |
| **Authority** | Create hypotheses; run research experiments on research data; request strategy packaging. |
| **Denied** | Access holdouts; declare validation success; deploy; claim live edge. |
| **Tools** | Stats stack (NumPy/SciPy/statsmodels), research backtest sandbox, hypothesis registry API. |
| **Memory** | Hypothesis lineages; failed mechanism explanations; feature graveyards. |
| **KPIs** | Hypotheses with explicit mechanism; replication rate; contamination incidents (target: 0). |
| **Comms** | `HypothesisProposed`, `ExperimentRegistered` → Strategy Engineering / Validation queue. |

---

### Agent 07 — Strategy Engineering Agent

| Field | Contract |
|-------|----------|
| **Mission** | Translate hypotheses into executable algorithms and reliable quantitative software; run code in isolation. |
| **Authority** | Produce versioned strategy packages; open PRs; run sandboxed builds/tests. |
| **Denied** | Production deploys; unrestricted code execution; research-plane credential use for trading. |
| **Tools** | Codegen+review workflow, sandbox runner, package builder, unit test harness. |
| **Memory** | Implementation patterns; sandbox failures; dependency risk notes. |
| **KPIs** | Build success rate; sandbox escape incidents (0); test coverage on strategy packages; mean time to package. |
| **Comms** | `StrategyPackageReady` with content hash → Validation. |

---

### Agent 08 — Independent Validation Agent

| Field | Contract |
|-------|----------|
| **Mission** | Adversarially invalidate strategies; prevent false discoveries; enforce programmatic gates. |
| **Authority** | Pass/fail validation; demand retests; freeze candidates on integrity breach. |
| **Denied** | Soft-pass on narrative grounds; alter gates without versioned protocol + approval; share holdout raw data to research. |
| **Tools** | Validation service (walk-forward, PBO, DSR, costs, regime tests), holdout accessor, report emitter. |
| **Memory** | Common failure modes; researcher gaming attempts; gate blind spots (for Meta-Learning proposals only). |
| **KPIs** | False discovery catch rate (audited); gate determinism; time-to-report; zero LLM-overridden outcomes. |
| **Comms** | Consumes `ValidationRequest`; emits `ValidationReport` (machine-readable). |

**Non-negotiable:** Gate outcomes are computed by code. LLM may explain a report; it may not change pass/fail.

---

### Agent 09 — Portfolio Management Agent

| Field | Contract |
|-------|----------|
| **Mission** | Combine independently validated strategies; manage correlation/regimes; allocate within risk constraints. |
| **Authority** | Propose allocations among **validated** strategies; request risk clearance. |
| **Denied** | Include unvalidated strategies; exceed risk limits; go live without authorization. |
| **Tools** | Covariance/regime tooling, optimizer (constrained), portfolio simulator. |
| **Memory** | Correlation breaks; allocation post-mortems. |
| **KPIs** | Realized vs target risk; diversification benefit; constraint violations (0). |
| **Comms** | `AllocationProposal` → Risk → (future) Execution. |

**Phase note:** Deferred until ≥2 validated strategies exist (or explicit single-strategy paper policy).

---

### Agent 10 — Risk Management Agent

| Field | Contract |
|-------|----------|
| **Mission** | Capital preservation: exposure, liquidity, leverage, drawdown, tail, operational risk. |
| **Authority** | **Independent veto** over deployment; tighten limits; trigger kill switch. |
| **Denied** | Weaken limits without human approval; validate scientific edge (that's Validation's job). |
| **Tools** | Limit registry, veto API, stress scenarios, kill switch. |
| **Memory** | Near-misses; limit breach drills; market halt events. |
| **KPIs** | Limit breach count; veto precision/recall vs later blowups; kill-switch drill success. |
| **Comms** | `RiskVeto`, `RiskClearance`, `KillSwitchActivated`. |

---

### Agent 11 — Execution and Monitoring Agent

| Field | Contract |
|-------|----------|
| **Mission** | Execute approved strategies; monitor integrity; measure TCA; detect live/paper deterioration. |
| **Authority** | Place orders only in allowed mode (paper first); pause on degradation signals per policy. |
| **Denied** | Trade unapproved strategies; access research secrets to “debug” live; raise risk limits. |
| **Tools** | Paper broker, (future) CCXT/exchange adapters, monitoring, TCA. |
| **Memory** | Venue outages; slippage regimes; incident runbooks. |
| **KPIs** | Slippage vs model; uptime; detection latency for alpha decay; unauthorized order attempts blocked. |
| **Comms** | `ExecutionReport`, `DegradationAlert` → Risk/CIO/Portfolio. |

---

### Agent 12 — Technology and Innovation Agent

| Field | Contract |
|-------|----------|
| **Mission** | Track AI/quant/infra innovations; prototype; benchmark vs institutional baseline; propose adoption. |
| **Authority** | Prototype in sandbox; submit benchmarked proposals. |
| **Denied** | Unilateral production adoption; introduce infra without demonstrated improvement. |
| **Tools** | Literature watch, prototype sandboxes, benchmark harness. |
| **Memory** | Evaluated tools with scores; rejected shiny objects. |
| **KPIs** | Benchmarked proposals/quarter; adoption win rate; complexity cost of adopted tools. |
| **Comms** | `InnovationProposal` → Meta-Learning / human ADR process. |

---

## 4. Communication Matrix (Summary)

| From → To | Typical messages |
|-----------|------------------|
| CIO → all | `TaskAssigned`, priority updates |
| Intelligence → Knowledge/Quant | `SourceIngested`, `EvidenceExtracted` |
| Data Eng → Research/Validation infra | `DatasetVersionPublished` |
| Quant → Strategy Eng | `HypothesisProposed` |
| Strategy Eng → Validation | `StrategyPackageReady` / `ValidationRequest` |
| Validation → CIO/Risk/Knowledge | `ValidationReport` |
| Portfolio → Risk | `AllocationProposal` |
| Risk → Execution/CIO | `RiskVeto` / `RiskClearance` / kill |
| Meta-Learning → all | `ImprovementProposal` |
| Tech Innovation → Meta/ADR | `InnovationProposal` |

---

## 5. Memory Architecture (Per Agent)

| Memory type | Store | Notes |
|-------------|-------|-------|
| Semantic | pgvector + registry docs | Scoped collections per agent |
| Episodic | Postgres event log | Tasks, decisions, outcomes |
| Procedural | Versioned playbooks/YAML | Change via improvement workflow |
| Reflective | Structured reflection rows | Required after failed tasks |

Retention and redaction policies: [SECURITY.md](./SECURITY.md).

---

## 6. KPI Governance

- KPI definitions live in versioned registries (`registries/kpis/*.yaml`).
- Agents **cannot** edit their KPI definitions.
- Meta-Learning may propose KPI changes; Independent Validation + human approve.
- Dashboards must show definition version next to scores.

---

## 7. Minimal Agent Set for Vertical Slice (M1)

Normative 13-step slice: [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) §4.

Runnable for M1 (not stubs):

1. **CIO / scheduler** — objective, delegation, resume (steps 1–2, 13)
2. **Intelligence** — real external research + source-backed evidence (steps 3–4)
3. **Data Engineering** — historical DatasetVersion (step 6); may be mostly deterministic
4. **Quantitative Research** — falsifiable hypothesis (+ consume next-experiment proposals)
5. **Strategy Engineering** — package + sandbox execution path (steps 7–8); human+tools OK initially
6. **Independent Validation** — deterministic validation service mandatory (step 9)
7. **Meta-Learning** — reflection, lessons, next-experiment proposal (steps 10–12); full improvement adoption loop still M4

Stub contracts only: Microstructure, Portfolio, Risk (veto API stub OK), Execution, Tech Innovation.

**Contradiction fixed:** An earlier draft omitted Intelligence and Meta-Learning; both are required by the authorized M1 slice.
