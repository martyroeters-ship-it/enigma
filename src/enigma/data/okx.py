"""OKX public funding-rate history, used only as a cross-venue sanity check (ADR-014).

OKX serves roughly the most recent three months of funding history; older ranges return
no rows. Pagination walks backwards with `after=<oldest fundingTime seen>`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from enigma.core.errors import ProviderUnavailableError
from enigma.data.http import PoliteHttp
from enigma.data.normalize import UTC_DT

OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate-history"
OKX_FUNDING_SCHEMA: dict[str, pl.DataType] = {
    "funding_time": UTC_DT,
    "funding_rate": pl.Float64(),
    "realized_rate": pl.Float64(),
}


def okx_swap_id(binance_symbol: str) -> str:
    """`BTCUSDT` -> `BTC-USDT-SWAP`."""
    for quote in ("USDT", "USDC", "USD"):
        if binance_symbol.endswith(quote) and len(binance_symbol) > len(quote):
            return f"{binance_symbol.removesuffix(quote)}-{quote}-SWAP"
    raise ValueError(f"cannot map {binance_symbol!r} to an OKX swap id")


def _ms(value: datetime) -> int:
    return int(value.astimezone(UTC).timestamp() * 1000)


class OkxFundingClient:
    def __init__(self, http: PoliteHttp | None = None, page_limit: int = 100) -> None:
        self.http = http or PoliteHttp()
        self.page_limit = page_limit

    @property
    def http_requests(self) -> int:
        return self.http.requests

    def fetch_funding_history(
        self, inst_id: str, start: datetime, end: datetime, max_pages: int = 50
    ) -> pl.DataFrame:
        """Funding events with `start <= funding_time < end`, sorted ascending."""
        start_ms, end_ms = _ms(start), _ms(end)
        rows: dict[int, tuple[float, float | None]] = {}
        after: int | None = None
        for _ in range(max_pages):
            params = {"instId": inst_id, "limit": str(self.page_limit)}
            if after is not None:
                params["after"] = str(after)
            response = self.http.get(OKX_FUNDING_URL, params=params)
            if response is None:
                raise ProviderUnavailableError(f"OKX funding history not found for {inst_id}")
            body = response.json()
            if body.get("code") != "0":
                raise ProviderUnavailableError(f"OKX error {body.get('code')}: {body.get('msg')}")
            page = body.get("data", [])
            if not page:
                break
            for item in page:
                ts = int(item["fundingTime"])
                realized = item.get("realizedRate")
                rows[ts] = (float(item["fundingRate"]), float(realized) if realized else None)
            oldest = min(int(item["fundingTime"]) for item in page)
            if oldest <= start_ms or (after is not None and oldest >= after):
                break
            after = oldest
        kept = sorted(ts for ts in rows if start_ms <= ts < end_ms)
        return pl.DataFrame(
            {
                "funding_time": [ts * 1000 for ts in kept],
                "funding_rate": [rows[ts][0] for ts in kept],
                "realized_rate": [rows[ts][1] for ts in kept],
            },
            schema={
                "funding_time": pl.Int64,
                "funding_rate": pl.Float64,
                "realized_rate": pl.Float64,
            },
        ).with_columns(pl.col("funding_time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))
