# ADR-014 — Binance public market-data archives as the M1 venue (resolves ADR-008 open item)

**Status:** Accepted (implementation team, 2026-11-25)

## Context
ADR-008 locks M1 to one venue with BTC/ETH perps (+ spot for basis) and asked for a venue with
clear historical-data access. Constraints: free, no account or API key, no dataset purchase.

Probed from the build environment on 2026-11-25:

| Endpoint | Result |
|----------|--------|
| `api.binance.com`, `fapi.binance.com` | HTTP 451 (geo-restricted from this host) |
| `data.binance.vision` (bulk archives) | 200 — monthly/daily zips for USD-M futures klines, funding rates, spot klines |
| `data-api.binance.vision` (public spot market-data REST) | 200 |
| `www.okx.com/api/v5/public/*` | 200 |
| `api.bybit.com` | 403 |

## Decision
- **Primary venue: Binance USD-M perpetuals (`BTCUSDT`, `ETHUSDT`) + Binance spot**, from the
  public `data.binance.vision` archives (CHECKSUM files verified) and `data-api.binance.vision`
  for recent spot bars. These are published by the exchange for public download without auth.
- **Secondary (cross-check only, not mixed into one dataset): OKX public REST** funding history
  for `BTC-USDT-SWAP` / `ETH-USDT-SWAP`, used for data-integrity cross-venue sanity checks.

## Rationale
Longest free history (2019+ funding), exchange-published checksums, highest perp liquidity
(lower microstructure confounding), and reachable from this host without workarounds.

## Consequences
- Datasets record exact archive URLs, checksums, and retrieval timestamps as provenance.
- Survivorship: only currently listed BTC/ETH contracts are used; noted on every DatasetVersion.

## Limitations
- Archives lag by about a day (daily files) and a month (monthly files); fine for research, not
  for live.
- Binance terms of use apply to the data; Enigma uses it for internal research only.
