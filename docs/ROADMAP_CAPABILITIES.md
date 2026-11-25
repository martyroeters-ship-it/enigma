# Capability roadmap: hybrid ML, narrative, calibration, postmortems, prediction markets, Jev decisions

**Status:** Scheduled after audit (2026-11-25, Europe/Istanbul)  
**Constraint:** Does **not** change active M2 acceptance criteria. Independent validation, deterministic risk, expenditure limits, and plane isolation stay as hard gates.

This document audits what Enigma already has, names the gaps, and schedules work by **research value × dependency readiness**. It deliberately avoids inventing parallel systems where Learning Architecture, Validation Protocol, or the strategy sandbox already cover the need.

---

## 1. Audit summary

| Capability | Already present | Gap (do not duplicate) |
|---|---|---|
| **Hybrid quantitative intelligence** | Strategy interface + sandbox + truncation/leakage (G01); methodology already says linear first, trees only after leakage; M2 strategy path allows a declared `ml` / `statistical` complexity class with mandatory simpler baseline; validation `benchmark_dominance` / complexity-must-earn-its-keep | No XGBoost/LightGBM/HMM/Bayesian/ensemble training loop; sandbox allowlist is numpy-centric; no evidence-driven model-class search or offline training artifact registry |
| **Narrative intelligence** | Intelligence plane + `Source`/`EvidenceItem` with retrieval timestamps and hashes; `SourceKind` includes `web_page`, `forum_post`; PIT bar panels + look-ahead gates | No first-class timestamped news/RSS/social ingest; no narrative→positioning→returns study protocol; no embargo aligned to publication/`available_at` for text |
| **Probability calibration** | Decision provenance + confidence fields on some paths; validation PSR/DSR (related, not the same); cost-model calibration called out as a microstructure KPI in specs | No durable forecast ledger; no Brier / log loss / reliability diagrams against realized outcomes; no systematic over/under-confidence detection for agents or strategies |
| **Decision postmortems** | Task `Reflection`, cycle lessons, abort lessons (L1), `ImprovementProposal` FSM; Learning Architecture requires propose → **independent** evaluate → adopt/rollback; M2 learning workstream targets `OutcomeRecord` / meta guidance / improvement experiments | Weak explicit split of **method error vs sampling variance**; postmortems not yet mandatory for every research/trading decision class; adoption still thin vs the full LEARNING_ARCHITECTURE cycle |
| **Prediction market intelligence** | Nothing implemented | No Polymarket/Metaculus/etc. adapter; no research-only evaluation design; trading those venues correctly deferred |
| **Structured decision provider (Jev)** | Chat `ModelGateway` + `StructuredLLM`; CIO deterministic priorizer; assessment in `docs/technology/jev-typesafe-assessment.md` | No provider-agnostic `DecisionProvider`; no M3-scale labeled eval vs heuristics/local/frontier LLMs; Jev not wired |

Specs already on the shelf (reuse, don’t rewrite):  
`QUANTITATIVE_METHODOLOGY.md` §7–8, `TECHNOLOGY_STACK.md` (LightGBM/PyTorch deferred), `LEARNING_ARCHITECTURE.md`, `VALIDATION_PROTOCOL.md` (leakage), `strategy/leakage.py`, abort/lesson paths.

---

## 2. Non-negotiables (apply to every item below)

1. Independent validation remains the only path to L3 / deployment clearance.  
2. Risk and spend ceilings stay deterministic code + registries — no model soft-pass.  
3. No look-ahead: text and PM data use explicit `available_at` ≤ decision time; G01-style checks extended where needed.  
4. Complexity (trees, HMM, ensembles, narrative features) must beat a registered simpler baseline on the **same** protocol.  
5. Active **M2 acceptance list is frozen**; these capabilities enter as later milestones or optional spikes that do not block M2.

---

## 3. Scheduling

```text
Now ── M2 (in flight) ──┬── M2.1 calibration + postmortem completion
                        ├── M3 hybrid ML + narrative foundations + DecisionProvider/Jev eval
                        ├── M4 narrative↔market studies + PM research feeds
                        └── Later: PM trading domain (separate charter)
```

### Priority order (research value given current dependencies)

1. **Probability calibration (M2.1)** — cheap, unlocks honest agent/strategy uncertainty; builds on existing decision records.  
2. **Decision postmortems (M2.1)** — completes institutional learning already half-built; required so “improvements” stay evidence-based.  
3. **Hybrid ML (M3)** — high value only after M2 codegen + leakage harness + baselines exist; sandbox/library policy is a hard dependency.  
4. **DecisionProvider + official TypeSafe Jev (M3 deliverable)** — structured decisions for high-volume classification/ranking/routing; integrate only if pre-registered thresholds are met.  
5. **Narrative intelligence (M3 foundations → M4 studies)** — high long-run value; needs dataset registry + PIT text store first; Jev may score narrative relevance once feeds exist.  
6. **Prediction market intelligence (M4 research-only → later trading)** — supplementary signal; trading is a separate product risk domain.

---

## 4. Work packages

### 4.1 Probability calibration — **M2.1**

**Goal:** Every probabilistic forecast Enigma emits (strategy probabilities, agent confidences, noul/score-style decisions) can be scored against realizations.

**Build**

- `ForecastRecord` (control/research as appropriate): subject, claim, probability, decided_at, available information refs, resolution time, outcome, scorer.  
- Metrics module: Brier, log loss, ECE/reliability bins; slice by agent, task_kind, family, regime.  
- Dashboard panel: calibration curves + over/under-confidence flags (systematic mean residual).  
- Wire: CIO/meta scores and any System-One-style decisions; optional strategy probability outputs when hypotheses declare them.

**Reuse:** `AgentDecision.model_calls` / execution_mode; validation statistics stay separate (PSR/DSR are gate math, not the forecast ledger).

**Acceptance:** ≥1 agent path and ≥1 strategy/hypothesis path resolve forecasts with Brier + reliability plot in the dashboard; overconfidence detector fires on a synthetic biased fixture.

### 4.2 Decision postmortems — **M2.1**

**Goal:** Successful and failed research (and later paper) decisions produce structured postmortems that separate **methodological error** from **ordinary variance**, with improvements only via independent evaluation.

**Build**

- Extend (don’t fork) `OutcomeRecord` / cycle reflection: fields `error_class ∈ {method, variance, infrastructure, data, policy}`, evidence refs, counterfactual test suggestion.  
- Classifier rules + optional LLM narrative labelled `execution_mode`; method-error claims remain ≤ L1 until an improvement experiment passes.  
- Hook every completed/aborted cycle (abort lessons already exist — merge taxonomies).  
- Improvement proposals **must** cite postmortem ids; adoption path = existing ImprovementProposal / MethodologyRegistry independent eval (LEARNING_ARCHITECTURE).

**Reuse:** `Reflection`, `Lesson`, abort lessons, `ImprovementProposal` FSM.

**Acceptance:** On the M1/M2 real-run history, postmortems label at least one method error (e.g. package-id collision, CPU-in-repro) vs variance-only validation FAILs; a proposed fix cannot flip to “adopted” without the independent benchmark harness.

### 4.3 Structured decisions: DecisionProvider + TypeSafe Jev — **M3 deliverable**

**Goal:** Add a provider-agnostic `DecisionProvider` beside the LLM gateway, benchmark official
TypeSafe Jev against deterministic heuristics, local models, and conventional LLMs, and integrate
Jev into operations **only** if it clears pre-registered thresholds with measurable gains.

**Prior art (do not redo):** [docs/technology/jev-typesafe-assessment.md](./technology/jev-typesafe-assessment.md)
— API shape, pricing, lookalike warning (`jevai.net`), gateway design sketch, and small local baselines.

**Hard rules**

- Endpoints: **`https://api.typesafe.ai` only** (`TYPESAFE_API_KEY`). Never `jevai.net` or resellers.  
- Jev **must not** override independent validation, deterministic risk, budget reservation/enforcement, or the kill switch (import-linter + tool contracts).  
- Does **not** block or interrupt M2; no M2 acceptance change.  
- Pin versioned model ids in production routes (`jev-1.14.0` etc.); do not rely on moving aliases.

**Build**

1. **`DecisionProvider` protocol** (alongside `ModelGateway` / `StructuredLLM`, not forced into chat completions):
   `decide(state, questions, *, model, budget, actor, task_id) -> DecisionResult` with choice / score / noul.  
   Implementations: `DeterministicDecisionProvider`, `LlmDecisionProvider` (schema via StructuredLLM),
   `TypeSafeDecisionProvider` (`POST /v1/systemone`).  
2. **Spend path:** same atomic budget ledger as LLM calls; verified pricing; provenance on `AgentDecision`.  
3. **Labeled evaluation set** (≥200 items, stratified; target ≥500 when feasible) covering:
   - classification; anomaly categorization; research-relevance scoring;
   - opportunity ranking; agent routing; high-volume narrative relevance/triage.  
4. **Baselines (required competitors):** deterministic heuristics; local Ollama; ≥1 conventional LLM when keys exist; Jev when `TYPESAFE_API_KEY` is authorized.  
5. **Metrics:** accuracy / macro-F1; probability calibration (Brier, log loss, ECE); latency; throughput; reliability (error/timeout rate); **cost per correct decision**.  
6. **Pre-register acceptance thresholds** in `registries/models/jev_m3_acceptance.yaml` *before* running the scored eval (freeze the label set hash). Suggested starting bars (tune and lock in that file):
   - macro-F1 ≥ best non-Jev baseline on the frozen set;
   - ECE no worse than the best calibrated baseline;
   - p50 latency ≤ 500 ms for ≤8 questions on ≤4k-token state *or* documented cost/latency trade accepted by CIO;
   - cost per correct decision ≤ 50% of the conventional LLM baseline on the same fan-out;
   - reliability ≥ 99% success on retries-included eval calls.  
7. **Integration iff thresholds met:** wire into CIO ranking assists, intelligence triage, anomaly labels, meta failure-class hints — never validation/risk/budget.  
8. **Institutional memory:** store `BenchmarkReport` + ADR adopt/reject; lessons from the eval; dashboard visibility under Models.

**Dependencies:** M2 gateway provenance + budgets; M2.1 calibration ledger (preferred for scoring Jev probabilities); narrative feed samples helpful but not required for the first stratified set.

**Acceptance (M3):** (a) `DecisionProvider` + three implementations land with tests; (b) frozen eval + pre-registered thresholds committed; (c) full benchmark matrix executed and archived; (d) adopt **or** explicit reject ADR with numbers; (e) if adopted, deny-list contracts prove validation/risk/budget cannot call Jev.

### 4.4 Hybrid quantitative intelligence — **M3**

**Goal:** Evidence-driven discovery/training/benchmarking of statistical and ML models (including XGBoost, LightGBM, Bayesian methods, HMMs, ensembles) without predetermined model-class favoritism.

**Build**

- `ModelClassRegistry`: candidates (linear, GLM, Bayesian linear, HMM, GBDT, ensembles…) with dependency packages, sandbox policy, and default complexity cost.  
- Offline fit runs as **versioned training jobs** on research-plane data only; artifacts content-hashed; inference code shipped into strategy packages.  
- Expand sandbox allowlist **per profile** (e.g. `ml-gbdt`) with pinned wheels — never open network or arbitrary pip.  
- Search policy: start from mechanism + baselines; promote complexity only if holdout protocol + baseline gate pass (already required by methodology).  
- Benchmark matrix: same validation protocol; record train CPU in cost ledger.

**Dependencies:** M2 strategy codegen + baseline comparison; leakage/truncation; dataset registry; atomic budgets.

**Out of scope for M3:** unconstrained PyTorch/RL (still deferred per TECHNOLOGY_STACK).

**Acceptance:** At least two model classes compete on one registered hypothesis; winner selected by validation metrics + baseline dominance, not by prior preference; failing class leaves FailureConditions.

### 4.5 Narrative intelligence — **M3 foundations, M4 studies**

**Goal:** Ingest timestamped public narrative sources and test measurable links to expectations, positioning, and subsequent returns — without leakage.

**M3 foundations**

- Adapters: RSS / public news APIs / allowlisted social (ToS-respecting, no auth bypass).  
- `NarrativeEvent` with `event_time`, `available_at` (publish/index time), source hash, topics.  
- Join rules to bars: feature at bar `t` may use events with `available_at ≤ t` only; extend quality checks + optional G01-style text truncation tests.  
- Intelligence planner actions: `ingest_feed`, `embargo_check`.

**M4 studies**

- Pre-registered hypotheses linking narrative intensity/polarity/surprise to funding, basis, OI, returns.  
- Explicit expectations/positioning proxies (funding, OI, perp premium) as co-variates.  
- Leakage red team: shuffled timestamps must kill the edge.

**Reuse:** KnowledgeBase provenance; DataCatalog PIT patterns; no second “news database” outside versioned datasets.

**Acceptance (M3 foundations):** End-to-end ingest of ≥1 real public feed into a PIT-safe feature column used in a dry research cycle.  
**Acceptance (M4):** One registered narrative hypothesis through full validation; leakage scramble test documented.

### 4.6 Prediction market intelligence — **M4 research feed; later trading charter**

**Goal:** Evaluate prediction markets as a **supplementary research input** for crypto/macro; do **not** conflate with trading those markets.

**M4 research-only**

- Read-only adapters (e.g. public Polymarket/Metaculus-style APIs where legitimate).  
- `PredictionMarketSnapshot` with market id, question, probability, volume, `asof`/`available_at`.  
- Research uses: calibration benchmarks for Enigma’s own forecasts; macro/crypto event priors as evidence (L0) for Intelligence/Quant — never as validation soft-pass.  
- Study: PM probability changes vs subsequent BTC/ETH moves / funding — pre-registered, PIT-safe.

**Later (separate domain)**

- Any **trading** of prediction markets = new execution venue + risk policy + legal review. New ADR and milestone; not an Enigma M4 add-on.

**Acceptance (M4):** Live read of ≥1 public PM source into knowledge/research; one calibration or association study registered; trading paths remain unimplemented and config-blocked.

---

## 5. Explicit non-goals / anti-duplication

| Do not build | Use instead |
|---|---|
| Second validation engine for ML | Existing `ValidationService` + protocol gates |
| Parallel “learning OS” | `LEARNING_ARCHITECTURE` + ImprovementProposal lifecycle |
| Unbounded web scrape platform | Intelligence HttpClient + allowlists + robots |
| Chatbot postmortems as truth | Structured OutcomeRecord + independent eval |
| PM trading “because the data is there” | Separate future charter |

---

## 6. Tracking

| ID | Capability | Earliest milestone | Blocks M2? |
|----|------------|--------------------|------------|
| CAP-CAL | Probability calibration | M2.1 | No |
| CAP-PMORT | Decision postmortems | M2.1 | No |
| CAP-HML | Hybrid ML / statistical models | M3 | No |
| CAP-JEV | DecisionProvider + official TypeSafe Jev eval/integrate | M3 (**deliverable**) | No |
| CAP-NAR | Narrative intelligence | M3 → M4 | No |
| CAP-PMR | Prediction market research feeds | M4 | No |
| CAP-PMT | Prediction market trading | Later / separate | No |

Progress updates go in milestone reports under the Project store `docs/milestones/`. Implementation PRs must reference the CAP- id above.
