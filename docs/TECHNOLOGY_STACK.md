# Enigma — Technology Stack

**Status:** Draft for review  
**Product:** Enigma  
**Method:** Each candidate is **Adopt / Defer / Reject** for M0–M1 unless noted, with rationale against requirements.

---

## 1. Principles

- Prefer Python core  
- Prefer simple explicit FSM over complex orchestration  
- Do not add infra for sophistication theater  
- Provider-independent model + exchange interfaces  

---

## 2. Evaluations

| Technology | Decision | Rationale |
|------------|----------|-----------|
| **Python 3.12+** | **Adopt** | Core language for quant, agents, validation |
| **FastAPI** | **Adopt (control/UI API)** | Typed HTTP for ui-api & internal promotion APIs; thin |
| **Pydantic v2** | **Adopt** | Boundary schemas; JSON Schema export |
| **PostgreSQL** | **Adopt** | Registry, FSM, outbox, audit |
| **pgvector** | **Defer (M2+)** | Useful for semantic memory; not required for M1 provenance |
| **DuckDB** | **Adopt** | Local/analytic queries over Parquet |
| **Parquet** | **Adopt** | Versioned columnar market data |
| **Polars** | **Adopt** | Fast dataframes for pipelines |
| **NumPy / SciPy** | **Adopt** | Numerics |
| **Statsmodels** | **Adopt as needed** | Classical stats; import when gates/research need |
| **VectorBT** | **Defer decision (ADR-003)** | Fast vectorized backtests; may fight event-time/funding realism — keep behind `BacktestEngine` interface |
| **NautilusTrader** | **Defer (M2+)** | Powerful event-driven; heavier ops/learning cost for M1 |
| **CCXT** | **Adopt (data/paper later)** | Exchange I/O abstraction; M1 historical may use CCXT fetch-OHLCV or flat files — interface still CCXT-shaped |
| **Playwright** | **Defer** | Use for JS-heavy sources only; prefer plain HTTP first |
| **Crawl4AI** | **Defer** | Evaluate vs httpx+readability; ToS risk; not on M1 critical path if one allowlisted fetch works |
| **MLflow** | **Defer** | Experiment tracking: start with registry tables; revisit when cardinality hurts |
| **LightGBM** | **Defer** | After linear baselines + leakage harness |
| **PyTorch** | **Defer** | High complexity; M3+ if ever |
| **Next.js** | **Defer UI polish** | Optional for M1; FastAPI+CLI sufficient; Adopt thin Next when dashboard stream staffed |
| **Docker** | **Adopt** | Plane-separated compose |
| **Redis/NATS** | **Defer** | Outbox+Postgres first |
| **LangGraph / AutoGen / crew-style** | **Reject for M1** | Explicit FSM preferred; revisit only with benchmark |
| **Neo4j** | **Reject for M1** | Postgres relationships enough |
| **Kubernetes** | **Reject for M1** | Compose first |

---

## 3. AI model interface

- Unified `LLMProvider` protocol: `complete`, `embed` (embed optional until pgvector)  
- Configured via env per plane; research may use models; **validation gate path does not call LLM for decisions**  
- Interchangeable vendors (OpenAI-compatible, Anthropic, local) behind adapter  

---

## 4. Recommended M1 lock list

**Must:** Python, Pydantic, Postgres, Parquet, Polars, NumPy, Docker, FastAPI (minimal), httpx (Intelligence fetch), CCXT or equivalent OHLCV source.

**Interface-hidden:** Backtest engine (VectorBT or custom).

**Stub OK:** Next.js, pgvector, Crawl4AI, MLflow, LightGBM, PyTorch, NautilusTrader.

---

## 5. Challenges to directive list

- **Crawl4AI + Playwright both** on day one is overkill — pick HTTP allowlist fetch for M1 step 3.  
- **MLflow + full knowledge graph** duplicates experiment metadata — registry tables win first.  
- **NautilusTrader + VectorBT** simultaneous = two engines; pick one behind interface.  

---

## 6. Open uncertainties

1. Package manager: **uv** recommended tentatively (speed, lockfile) — confirm ADR-001.  
2. Exact OHLCV source license/ToS for historical BTC/ETH.  
3. Whether paper trading uses CCXT sandbox or internal simulator first (recommend internal simulator, then CCXT).
