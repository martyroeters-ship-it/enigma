"""Instrument constructors for the M1 universe (ADR-008/ADR-014: BTC/ETH on Binance)."""

from __future__ import annotations

from enigma.schemas.common import Instrument

PERP_VENUE = "binance_usdm"
SPOT_VENUE = "binance_spot"
SUPPORTED_SYMBOLS: dict[str, tuple[str, str]] = {
    "BTCUSDT": ("BTC", "USDT"),
    "ETHUSDT": ("ETH", "USDT"),
}


def _split(symbol: str) -> tuple[str, str]:
    try:
        return SUPPORTED_SYMBOLS[symbol]
    except KeyError as exc:
        raise ValueError(
            f"{symbol!r} is outside the M1 universe {sorted(SUPPORTED_SYMBOLS)}"
        ) from exc


def binance_perp(symbol: str) -> Instrument:
    base, quote = _split(symbol)
    return Instrument(symbol=symbol, base=base, quote=quote, kind="perpetual", venue=PERP_VENUE)


def binance_spot(symbol: str) -> Instrument:
    base, quote = _split(symbol)
    return Instrument(symbol=symbol, base=base, quote=quote, kind="spot", venue=SPOT_VENUE)
