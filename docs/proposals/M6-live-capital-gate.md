# M6 specification — live-capital authorization gate

**Status:** proposed. This document does not authorize live trading, real capital, or a broker.
**Depends on:** Constitution prohibitions 9 and the capital rule, `RiskLimits.load` in
`src/enigma/risk/service.py`, `registries/risk/limits.yaml`, the human risk acknowledgement
in `src/enigma/risk/acknowledge.py`, and the M5 host boundary.

## Current control

`registries/risk/limits.yaml` sets `live_trading_enabled: false`. `RiskLimits.load` raises
`ValueError` if that flag is true. Paper trading is a separate flag and is also false in the
production file. A paper session still requires a validation PASS, a risk clearance, and a
human acknowledgement. An agent cannot satisfy those by editing this specification.

## Gate

Live capital becomes possible only when every line below is true. Missing any line keeps the
flag false and keeps the loader refusal in place.

1. A named human, not an agent role, records an authorization document with the account, the
   venue, the maximum notional, and the expiry. The document is stored outside any agent
   writable plane.
2. Constitution text that forbids live capital is amended by that human. An agent edit does
   not count.
3. M5 is implemented: the order path cannot read the holdout store, and validation cannot
   call a model or a decision provider.
4. Independent validation has PASSed the specific strategy package under the protocol the
   human names. A research-window result, a fixture summary, or an agent-written PASS does
   not count.
5. Risk clearance for that package is a `clearance` from the risk service under limits the
   human signed. The service still refuses a file with `live_trading_enabled: true` until
   the human replaces the loader rule in a reviewed change.
6. The order adapter does not exist in this repository until the change that adds it is
   reviewed against this list. This specification does not add that adapter.

## What an agent cannot do

- An agent cannot mark this gate passed, set `live_trading_enabled`, or copy
  `registries/risk/limits-paper-enabled.yaml` over the production limits.
- An agent cannot treat a green test, a paper fill, or a dashboard state as authorization.
- An agent cannot store the authorization document in the research or execution plane.

## Acceptance

M6 is not accepted by merging this file. It is accepted only when a human records the
authorization in item 1 and the loader change in item 5, and a reviewer checks items 2–4
against artifacts that human did not ask an agent to invent. Until then, live trading stays
disabled.
