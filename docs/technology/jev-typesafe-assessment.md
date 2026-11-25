# Technology assessment: Jev (TypeSafe AI) as a structured-decision provider

**Status:** assessed, not adopted  
**Date:** 2026-11-25 (Europe/Istanbul)  
**Assessor:** Enigma implementation (M2 parallel investigation)  
**Recommendation:** Do **not** integrate into the M2 critical path. Revisit after M2 with a TypeSafe early-access key and a labeled Enigma evaluation set. Prefer the official TypeSafe API only; treat `jevai.net` as unofficial.

## 1. What was investigated

| Claimed product | Finding |
|---|---|
| **Jev** | TypeSafe AI’s “System One” decision model: returns typed decisions and calibrated probabilities, not free-form text. |
| **Official maker** | [typesafe.ai](https://typesafe.ai), docs at [docs.typesafe.ai](https://docs.typesafe.ai), API at `https://api.typesafe.ai`. |
| **`jevai.net`** | Presents itself as Jev’s home page with a different price ($0.084/MTok) and contact `hello@jevai.net`. Independent reporting (TrustList, RateLimitNow) flags it as a post-launch lookalike, not TypeSafe. **Do not send Enigma credentials or research data there.** |
| **`jevai.ai` / `jevai.info`** | Community / independent resources that point back to TypeSafe for keys. |

This assessment evaluates **TypeSafe’s official Jev** as the only legitimate integration candidate. References to “jevai.net” below are warnings only.

## 2. Official API and contract

Verified live (2026-11-25):

- `POST https://api.typesafe.ai/v1/systemone` exists (HTTP 405 on GET with `Allow: POST`).
- Auth: `Authorization: Bearer $TYPESAFE_API_KEY`.
- Request shape (OpenAPI 0.2.0): `{ model, state, questions, assets? }` — **not** an OpenAI chat-completions clone.
- Question primitives: `choice` (≤255 options), `score` (2–10 levels), `noul` (yes-probability), `box` (image, experimental, Jev 1.14).
- Many questions share one `state` and are evaluated in parallel (fan-out).
- Response includes per-question answers, probabilities/confidence, versioned `model` id, and token usage.
- Model IDs: `jev-1.14.0`, `jev-1.13.0`; aliases `jev-latest` / `jev-preview` (both currently → 1.14.0).
- Context: 64k per-request / 32k per-question; optional long-context header → 262k (slower, less accurate).

**Blocked here:** no `TYPESAFE_API_KEY` in this environment, so no live accuracy/latency/reliability measurement against TypeSafe.

## 3. Pricing, limits, privacy

| Item | Official (TypeSafe) | Notes |
|---|---|---|
| Price | **$0.042 / million input tokens**; output free | Same for 1.13 and 1.14. Prepaid credits; 402 when balance empty. |
| Default rate limits | **100k tokens/s**, **80 requests/s** per org | Higher limits via credit prepayment / request form. |
| Training on customer data | Docs state **not trained** on requests/responses; no per-customer fine-tune | Shape behaviour via `state` + question `instructions`/`criteria`. |
| ZDR | Enterprise zero data retention offered | Details under Legal / support form. |
| `jevai.net` list price | $0.084 / MTok | Unofficial; ignore for planning. |

Cost implication for Enigma: at $0.042/MTok, scoring ~2k tokens of evidence+hypothesis state with 8 parallel questions ≈ **$0.000084** per fan-out. That is cheap versus frontier chat models, and competitive with “free” local Ollama only when operator time and latency matter more than GPU/CPU.

## 4. Fit to Enigma workloads

| Workload | Fit | Why |
|---|---|---|
| High-volume classification (anomaly class, failure class, source type) | **Strong candidate** | Native `choice` + probabilities; batch questions on one state. |
| Scoring / opportunity ranking / research prioritization | **Strong candidate** | Native `score` / multi-`noul`; can feed CIO priorizer as a soft signal. |
| Information extraction (horizon, columns, gate names) | **Partial** | Constrained choice/score works; free-form field extraction is not the product. Prefer schemas via existing `StructuredLLM`. |
| Market anomaly categorization | **Good** | Same as classification over fixed taxonomies. |
| Agent routing | **Good** | `choice` over roles/tools with confidence thresholds. |
| Complex quantitative hypothesis generation / math / codegen | **Poor — out of scope** | Jev does not generate text, proofs, or code. Keep frontier/local generative models. |
| Independent validation, budget enforcement, risk vetoes | **Forbidden use** | Must remain deterministic. Jev must never set `overall`, spend ceilings, or trading clearance. |

## 5. Benchmarks (what we could measure)

### 5.1 Local baselines on Enigma-shaped tasks (measured)

Five representative cases (anomaly classification, research routing, information-value score, evidence-support noul, horizon extraction). Results saved in this investigation’s summary artifact.

| Provider | Accuracy | Mean latency | Tokens | Cost |
|---|---:|---:|---:|---:|
| Deterministic heuristics | 1.00 | ~0.01 ms | 0 | $0 |
| Ollama `qwen2.5:3b` via Enigma gateway | 0.60 | ~3.4 s | 766 | $0 |
| Ollama `qwen2.5:7b-instruct` via Enigma gateway | 0.80 | ~4.5 s | 569 | $0 |
| TypeSafe Jev | **not run** | — | — | — |

Caveats:

- The deterministic score is on a tiny, hand-authored set where keywords match the label; it is a lower bound on engineering cost for *known* taxonomies, not a claim of general superiority.
- Local LLMs are slow on this CPU host; Jev’s published cookbooks claim large speedups from question fan-out, but that is **not independently verified here**.
- Without a TypeSafe key, we cannot verify calibration (Brier / reliability diagrams) on Enigma labels.

### 5.2 Published TypeSafe evidence (third-party / vendor)

- Parallel-questions cookbook: batching N questions on one GDPR article claimed ~12× cheaper and ~12× faster than N single-question calls, with answer stability (run-to-run std ≤ ~0.015).
- Training claim: RLCD for calibrated probabilities (group-level calibration, not per-decision guarantees).

Treat vendor cookbooks as directional until Enigma reproduces them on institutional data.

## 6. Gateway integration design (if adopted later)

**Do not** force Jev into the chat `ModelRequest` path. The wire format is a different product surface.

Recommended shape (minimal lock-in):

```text
DecisionProvider (protocol)
  .decide(state, questions, *, model, budget, actor, task_id) -> DecisionResult

TypeSafeDecisionProvider  # POST /v1/systemone
DeterministicDecisionProvider  # existing classifiers / CIO scores
LlmDecisionProvider  # thin adapter over StructuredLLM for choice/score/noul schemas
```

Wiring rules:

1. Register under `registries/models/` as provider kind `typesafe` / `system_one` with `api_key_env: [TYPESAFE_API_KEY]`, verified pricing, and spend via the **same atomic budget ledger** as LLM calls.
2. Expose a small `DecisionGateway` (or `StructuredLLM.decide`) used by CIO prioritization, intelligence gap tagging, anomaly labels, and meta failure-class hints.
3. Record `execution_mode`, model version (`jev-1.14.0`), request/response hashes, and confidence on `AgentDecision` — same provenance path as generative calls.
4. **Hard deny list:** `enigma.validation`, risk clearance, budget `authorize`/`reserve`, kill switch, holdout seal.
5. Keep generative hypothesis/strategy codegen on Ollama/OpenAI/Anthropic routes unchanged.
6. Pin versioned model IDs in production routes; aliases may move.

Complexity cost: one new adapter + protocol (~small), not a gateway rewrite. Vendor lock-in risk is low if callers depend only on `DecisionProvider` and can fall back to deterministic/`StructuredLLM`.

## 7. Adoption criteria (pre-declared)

Adopt a TypeSafe spike into main **only if** all of the following hold on a frozen Enigma label set (≥100 items spanning routing, anomaly class, evidence support, abort classification):

1. **Quality:** macro-F1 ≥ local `qwen2.5:7b-instruct` **and** ≥ deterministic baseline on the subset that heuristics cover, **or** clear win on the subset heuristics miss.
2. **Calibration:** expected calibration error competitive with (not worse than) local LLM softmax proxies on the same labels.
3. **Latency:** p50 ≤ 500 ms for ≤8 questions on ≤4k-token state (or documented reason to accept slower).
4. **Cost:** measured $/decision ≤ 50% of a frontier structured call for the same fan-out, after budget accounting.
5. **Safety:** import-linter / tool contracts prove validation and risk cannot call it; kill switch and spend cap still bind.
6. **Provenance:** every decision stores versioned model id + hashes; `jev-latest` unused in production routes.

Until then: **no dependency, no credentials in CI, no replacement of M2 generative work.**

## 8. Relation to M2 priorities

This investigation does **not** change M2 sequencing. Parallel workers continue on atomic budgets, autonomous hypotheses, adaptive intelligence, datasets, holdout integrity, and meta-learning. Jev is a candidate *accelerator* for high-volume typed decisions after those foundations land.

## 9. Credentials (if a spike is authorized later)

| Variable | Purpose |
|---|---|
| `TYPESAFE_API_KEY` | Official TypeSafe console key for `api.typesafe.ai` |
| Never | Keys or traffic to `jevai.net` / unidentified resellers |

Also keep `ENIGMA_MAX_DAILY_API_SPEND_USD` as the hard ceiling (default 0 until raised).

## 10. Sources

- https://docs.typesafe.ai/models  
- https://docs.typesafe.ai/api-reference (OpenAPI System One)  
- https://docs.typesafe.ai/legal  
- https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost  
- https://docs.typesafe.ai/cookbooks/parallel_questions  
- https://docs.typesafe.ai/introduction/machine-learning-primer  
- Lookalike analysis: TrustList / RateLimitNow public write-ups on `jevai.net`  
- Local baseline: `/tmp/jev-eval/summary.json` (copied beside this assessment in the Project store)
