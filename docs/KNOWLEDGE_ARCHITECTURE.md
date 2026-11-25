# Enigma — Knowledge Architecture

**Status:** Draft for review  
**Product:** Enigma  
**Goal:** Persistent institutional memory with provenance, uncertainty, and explicit claim levels — not a dumping ground for LLM text.

---

## 1. Design Principles

1. **Provenance first** — every node/edge records source, time, actor, method.
2. **Claim levels are explicit** — never blur unverified vs validated edge.
3. **Failures are first-class** — failed experiments remain queryable forever.
4. **Append-oriented** — corrections create new versions; no silent rewrite of history.
5. **Start relational** — graph/vector layers earn their keep via retrieval metrics ([ARCHITECTURE.md](./ARCHITECTURE.md) challenge).

---

## 2. Claim Levels

| Level | Name | Meaning | Who can write |
|-------|------|---------|---------------|
| L0 | Unverified information | Extracted or asserted; not empirically tested by Enigma | Intelligence, humans |
| L1 | Theoretical hypothesis | Falsifiable mechanism + predicted effect | Quant Research |
| L2 | Historically validated statistical relationship | Passed research-plane tests only (not independent holdout) | Research services |
| L3 | Independently validated edge | Passed Independent Validation protocol on holdout/paper rules | Validation service only |

**Promotion rules:** L0→L1 by hypothesis registration; L1→L2 by research experiment records; L2→L3 **only** via `ValidationReport` with pass. Downgrades create new claim versions with reason codes.

---

## 3. Entity Graph (Logical)

Connected entity types:

- Source, Author, Publication
- MarketMechanism
- MathematicalModel
- Dataset (+ DatasetVersion)
- Hypothesis
- StrategyImplementation (package)
- Experiment
- ValidationResult
- MarketRegime
- FailureCondition
- AgentDecision
- TechnologicalInnovation
- ImprovementProposal

### Core relationships

```text
Source --cites--> Publication
Publication --discusses--> MarketMechanism | MathematicalModel
Hypothesis --based_on--> Publication | MarketMechanism | DatasetVersion
Hypothesis --tested_by--> Experiment
Experiment --uses--> DatasetVersion + StrategyImplementation
Experiment --produces--> ResearchMetrics (non-L3)
StrategyImplementation --validated_by--> ValidationResult
ValidationResult --under_regime--> MarketRegime
ValidationResult --failure_modes--> FailureCondition
AgentDecision --references--> Hypothesis | ValidationResult | ImprovementProposal
ImprovementProposal --targets--> Agent | Methodology | Tool
```

---

## 4. Data Schemas (Initial)

Schemas are normative for `enigma-schemas`. Fields marked ★ are required.

### 4.1 DatasetVersion

```yaml
id: uuid ★
name: string ★
asset_scope: [BTC-PERP, ETH-PERP, ...] ★
venue: string ★
schema_version: semver ★
time_range: {start: datetime, end: datetime} ★
partitioning: string
content_hash: sha256 ★
pit_certified: bool ★
survivorship_notes: string
quality_report_id: uuid
created_by: agent_id ★
created_at: datetime ★
```

### 4.2 Hypothesis

```yaml
id: uuid ★
title: string ★
mechanism: string ★          # causal/structural explanation required
prediction: string ★
claim_level: L1 ★
instruments: [string] ★
horizon: string ★
dataset_version_ids: [uuid]
source_ids: [uuid]
contradictory_evidence_ids: [uuid]
uncertainty_notes: string
created_by: agent_id ★
methodology_version: string ★
created_at: datetime ★
```

### 4.3 Experiment

```yaml
id: uuid ★
hypothesis_id: uuid ★
code_sha: string ★
config_hash: sha256 ★
seed: int
dataset_version_id: uuid ★
status: enum(RUNNING, SUCCEEDED, FAILED, ABORTED) ★
metrics: object                 # research-plane only
artifact_uri: string
started_at / ended_at: datetime
```

### 4.4 StrategyPackage

```yaml
id: uuid ★
hypothesis_id: uuid ★
experiment_ids: [uuid] ★
entrypoint: string ★
content_hash: sha256 ★
interface_version: semver ★
cost_model_version: string ★
sandbox_profile: string ★
```

### 4.5 ValidationReport

```yaml
id: uuid ★
strategy_package_id: uuid ★
protocol_version: string ★
holdout_dataset_ids: [uuid] ★
gates:                          # each gate: {name, params, result: PASS|FAIL, stat, threshold}
  - name: string ★
    result: enum ★
    details: object
overall: enum(PASS, FAIL) ★     # computed, not LLM
deflated_sharpe: float | null
pbo: float | null
cost_assumptions_hash: sha256 ★
created_at: datetime ★
validator_build_sha: string ★
```

### 4.6 ImprovementProposal

```yaml
id: uuid ★
target: string ★
change_summary: string ★
success_metric: string ★
baseline_metric_value: number
proposed_by: agent_id ★
status: enum(PROPOSED, BENCHMARKING, APPROVED, REJECTED, ROLLED_OUT, REVERTED)
```

### 4.7 RunObjective / Reflection / Lesson / NextExperiment (M1)

```yaml
# RunObjective
id: uuid ★
statement: string ★
market_scope: [BTC, ETH, ...] ★
process_success_criteria: string ★   # not “make money”
created_at: datetime ★

# Reflection
id: uuid ★
run_id: uuid ★
agent_id: string ★
outcome: enum(success, failure, aborted) ★
what_worked: string ★
what_failed: string ★
suspected_systemic_issue: string | null
created_at: datetime ★

# LessonRecord
id: uuid ★
source_run_id: uuid ★
insight: string ★
evidence_refs: [uuid] ★
created_by: agent_id ★

# NextExperimentProposal
id: uuid ★
parent_run_id: uuid ★
lesson_ids: [uuid] ★
hypothesis_delta: string ★
rationale_refs: [uuid] ★
```

### 4.8 Message envelope

See [ARCHITECTURE.md](./ARCHITECTURE.md) §5.2. Payload schemas versioned alongside message_type.

---

## 5. Storage Layout

| Concern | Store | Notes |
|---------|-------|-------|
| Registry + FSM + audit | PostgreSQL | Source of truth for IDs and states |
| Semantic retrieval | pgvector (optional M2+) | Embeddings of docs/hypotheses; not evidence of truth |
| Analytics / research frames | Parquet + DuckDB | Columnar, versioned paths |
| Large artifacts | Object store (FS/MinIO/S3) | Backtest outputs, packages |
| Agent episodic/reflective | PostgreSQL | Tied to agent_id + task_id |

**Path convention (research store):**

```text
s3://enigma-research/datasets/{name}/{version}/...
s3://enigma-research/artifacts/{experiment_id}/...
```

Validation store uses a separate bucket/account: `enigma-validation://...`.

---

## 6. Uncertainty & Contradiction

- Every L0/L1 node may link `supports` and `contradicts` edges with weights in `[0,1]` and method tags (`citation`, `replication_fail`, etc.).
- Validation failures automatically attach `FailureCondition` nodes (overfit, leakage, cost fragility, …).
- UI/API must show contradictions adjacent to claims; hiding them is a product defect.

---

## 7. Knowledge Access Control

| Plane | Read | Write |
|-------|------|-------|
| Research | Research registry + L0–L2 | L0–L2 per agent rules |
| Validation | Validation registry + necessary package metadata | L3 ValidationResult only |
| Execution | Approved deployment manifests | Execution events only |

Holdout raw series never appear in research semantic search indexes.

---

## 8. Retention

- Experiments & validation reports: **retain forever** (or legal max with offline cold storage).
- Raw tick data: retention policy TBD per cost (open uncertainty).
- Agent reflections: retain ≥ 24 months online.

---

## 9. Open Uncertainties

1. Graph DB (e.g., Neo4j) vs Postgres recursive CTEs — defer until query patterns hurt.
2. Embedding model choice and re-embed policy on model change.
3. Multilingual source canonicalization strategy.
4. Legal/ToS constraints on crawling specialized forums (policy before tooling).
