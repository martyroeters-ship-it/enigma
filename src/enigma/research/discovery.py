"""Formulate a research question from evidence the institution already holds.

The four registered mechanisms are seeds. A new mechanism is registered only when
a verbatim claim names one measurable column, states one direction, and is not a
restatement of a seed. A claim that needs data the panel does not have becomes a
data request, not a hypothesis. Nothing is invented to keep the backlog busy.
"""

from __future__ import annotations

import re
from datetime import datetime

from enigma.core.hashing import sha256_hex
from enigma.data.panel import PANEL_COLUMNS
from enigma.research.mechanisms import _ZSCORE_HELPER, MECHANISMS, Mechanism
from enigma.schemas.common import AgentRole, Instrument
from enigma.schemas.knowledge import EvidenceItem
from enigma.schemas.market import MarketScanReport
from enigma.schemas.quant import DatasetAcquisitionRequest
from enigma.schemas.research import DiscoveredMechanism, ResearchObjective
from enigma.store import Stores

_HORIZONS = ("1h", "8h", "1d")
_POSITIVE = (
    "continuation",
    "continues",
    "momentum",
    "leads",
    "followed by higher",
    "higher subsequent",
    "positively associated",
    "positive association",
)
_NEGATIVE = (
    "reversal",
    "reverts",
    "mean reverts",
    "followed by lower",
    "lower subsequent",
    "negatively associated",
    "negative association",
)
# Names in prose for columns the research panel actually stores.
_COLUMN_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("perp_quote_volume", ("perp_quote_volume", "quote volume")),
    ("perp_volume", ("perp_volume", "traded volume")),
    ("perp_close", ("perp_close", "perpetual price", "perp price")),
    ("spot_close", ("spot_close", "spot price")),
    ("funding_rate", ("funding_rate", "funding rate")),
    ("basis", ("basis",)),
)
# Public series the panel does not contain. These become data requests.
_MISSING_PHRASES: tuple[tuple[str, str], ...] = (
    ("open interest", "open_interest"),
    ("liquidation", "liquidation_volume"),
    ("order flow", "order_flow_imbalance"),
    ("order-flow", "order_flow_imbalance"),
    ("sentiment", "sentiment_score"),
    ("cross-exchange", "cross_exchange_price"),
    ("lead-lag", "cross_asset_return"),
    ("cross-asset", "cross_asset_return"),
)
_ENTRYPOINT = "DiscoveredProbe"


def resolve_mechanism(documents: object, mechanism_id: str) -> Mechanism | None:
    """Seed library first, then a mechanism this institution formulated."""
    known = MECHANISMS.get(mechanism_id)
    if known is not None:
        return known
    getter = getattr(documents, "get", None)
    if getter is None:
        return None
    found = getter(DiscoveredMechanism, mechanism_id)
    if found is None:
        return None
    return _as_mechanism(found)


def replenish(stores: Stores, now: datetime, *, backlog_exhausted: bool = False) -> str | None:
    """Register one new objective.

    The caller decides that the viable backlog is empty. This function does not
    rank the backlog, so the research package does not depend on the orchestrator.
    """
    if not backlog_exhausted:
        return None
    pending = _pending_objective(stores)
    if pending is not None:
        return pending
    known_text = [f"{m.name} {m.formal_statement}" for m in MECHANISMS.values()]
    known_text.extend(
        f"{item.name} {item.formal_statement}"
        for item in stores.documents.list(DiscoveredMechanism, limit=500)
    )
    used = {
        evidence_id
        for item in stores.documents.list(DiscoveredMechanism, limit=500)
        for evidence_id in item.evidence_ids
    }
    for evidence in stores.documents.list(EvidenceItem, limit=2000):
        if evidence.id in used or evidence.is_ai_interpretation:
            continue
        text = f"{evidence.claim} {evidence.excerpt}"
        missing = _missing_column(text)
        column = _panel_column(text)
        if missing and column is None:
            _request_data(stores, evidence, missing, now)
            continue
        if column is None:
            continue
        sign = _sign(text)
        if sign is None or _too_close(text, known_text):
            continue
        horizon = _horizon(text)
        return _register(
            stores,
            now,
            column=column,
            sign=sign,
            horizon=horizon,
            claim=evidence.claim.strip(),
            evidence_ids=(evidence.id,),
            origin="evidence",
            symbol=_symbol(text),
            contributions={
                "intelligence": evidence.id,
                "quant_research": "falsifiable statement taken from the verbatim claim",
                "market_microstructure": "no separate observation was required",
                "cio": "objective opened because the seeded backlog had no viable item",
            },
        )
    return _from_fresh_anomaly(stores, now, known_text)


def _pending_objective(stores: Stores) -> str | None:
    used = {
        str(row["state"].get("objective_id") or "")
        for row in stores.checkpoints.list("research_cycle")
    }
    for item in stores.documents.list(DiscoveredMechanism, limit=100):
        if item.objective_id not in used:
            return item.objective_id
    return None


def _register(
    stores: Stores,
    now: datetime,
    *,
    column: str,
    sign: int,
    horizon: str,
    claim: str,
    evidence_ids: tuple[str, ...],
    origin: str,
    symbol: str,
    contributions: dict[str, str],
) -> str:
    digest = sha256_hex(f"{claim}|{column}|{sign}|{horizon}|{symbol}")
    mechanism_id = "disc_" + digest[:26]
    objective_id = "obj_" + digest[:26].upper()
    formal = (
        f"Let z_t be the trailing z-score of {column}. "
        f"H1: E[next {horizon} perp return | z_t > k] has sign {sign}. "
        "H0: that conditional mean is 0. The probe does not search the opposite sign."
    )
    prediction = (
        f"When {column} is extreme relative to its own trailing history, the next "
        f"{horizon} perpetual return tends to move with sign {sign}."
    )
    name = claim.split(".", maxsplit=1)[0].strip()[:120] or f"{column} probe"
    document = DiscoveredMechanism(
        id=mechanism_id,
        name=name,
        mechanism=claim,
        prediction=prediction,
        formal_statement=formal,
        features=(column,),
        horizon=horizon,
        evidence_ids=evidence_ids,
        entrypoint=_ENTRYPOINT,
        source_code=probe_source(column, mechanism_id),
        param_grid={"lookback": (21, 42), "z_entry": (1.5,), "sign": (float(sign),)},
        origin=origin,
        objective_id=objective_id,
        contributions=contributions,
        created_by=AgentRole.QUANT_RESEARCH,
        created_at=now,
    )
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    objective = ResearchObjective(
        id=objective_id,
        statement=(
            f"{claim} Pre-registered probe of {column} on {symbol} at {horizon}. "
            "This registers a falsifiable question. It is not a validation result."
        ),
        instruments=(
            Instrument(
                symbol=symbol,
                base=base,
                quote="USDT",
                kind="perpetual",
                venue="binance_usdm",
            ),
        ),
        topics=("discovered", mechanism_id, symbol, horizon, column),
        max_trials=2,
        created_by=AgentRole.CIO,
        created_at=now,
    )
    stores.documents.put(document, recorded_by=AgentRole.QUANT_RESEARCH.value)
    stores.documents.put(objective, recorded_by=AgentRole.CIO.value)
    stores.audit.record(
        AgentRole.CIO.value,
        "research.discovered",
        mechanism_id,
        {"objective_id": objective_id, "column": column, "origin": origin, "horizon": horizon},
    )
    return objective_id


def _from_fresh_anomaly(stores: Stores, now: datetime, known_text: list[str]) -> str | None:
    reports = stores.documents.list(MarketScanReport, limit=1)
    if not reports:
        return None
    report = reports[0]
    if report.objective_id or not report.top_field or report.top_available_at is None:
        return None
    if report.top_field not in PANEL_COLUMNS or report.top_field in {"funding_rate", "basis"}:
        return None
    if report.top_zscore is None or abs(report.top_zscore) < 2.5:
        return None
    claim = (
        f"The latest {report.top_symbol} {report.top_field} observation at "
        f"{report.top_available_at.isoformat()} has z={report.top_zscore:.3f}. "
        f"A positive association between {report.top_field} and the next return "
        "is the question. It has not been a registered mechanism."
    )
    if _too_close(claim, known_text):
        return None
    return _register(
        stores,
        now,
        column=report.top_field,
        sign=1,
        horizon="8h",
        claim=claim,
        evidence_ids=(),
        origin="observation",
        symbol=report.top_symbol or "BTCUSDT",
        contributions={
            "intelligence": "no new literature claim",
            "market_microstructure": report.id,
            "quant_research": "falsifiable statement from the latest unused anomaly",
            "cio": "objective opened because the seeded backlog had no viable item",
        },
    )


def probe_source(column: str, name: str) -> str:
    """One measurement instrument. The economic claim lives on the mechanism, not here."""
    return (
        "import math\n\nimport numpy as np\n\n"
        "from enigma.strategy.interface import Strategy, StrategyMeta\n"
        + _ZSCORE_HELPER
        + f"""
class DiscoveredProbe(Strategy):
    def configure(self, params):
        self.lookback = int(params["lookback"])
        self.z_entry = float(params["z_entry"])
        self.sign = float(params["sign"])

    def required_columns(self):
        return ({column!r},)

    def metadata(self):
        return StrategyMeta(name={name!r}, version="1")

    def target_positions(self, data):
        series = np.nan_to_num(np.asarray(data[{column!r}], dtype=float), nan=0.0)
        z = _rolling_z(series, self.lookback)
        pos = np.zeros(series.shape[0])
        pos[z > self.z_entry] = self.sign
        pos[z < -self.z_entry] = -self.sign
        return pos
"""
    )


def _as_mechanism(document: DiscoveredMechanism) -> Mechanism:
    return Mechanism(
        id=document.id,
        name=document.name,
        mechanism=document.mechanism,
        prediction=document.prediction,
        formal_statement=document.formal_statement,
        keywords=(),
        contra_keywords=(),
        features=document.features,
        param_grid={key: tuple(values) for key, values in document.param_grid.items()},
        entrypoint=document.entrypoint,
        source_template=document.source_code,
        horizon=document.horizon,
    )


def _panel_column(text: str) -> str | None:
    lowered = text.lower()
    found = [
        column
        for column, phrases in _COLUMN_PHRASES
        if any(phrase in lowered for phrase in phrases)
    ]
    if len(found) != 1:
        return None
    return found[0]


def _missing_column(text: str) -> str | None:
    lowered = text.lower()
    found = [column for phrase, column in _MISSING_PHRASES if phrase in lowered]
    if len(set(found)) != 1:
        return None
    return found[0]


def _sign(text: str) -> int | None:
    lowered = text.lower()
    positive = any(phrase in lowered for phrase in _POSITIVE)
    negative = any(phrase in lowered for phrase in _NEGATIVE)
    if positive == negative:
        return None
    return 1 if positive else -1


def _horizon(text: str) -> str:
    for horizon in _HORIZONS:
        if re.search(rf"\b{horizon}\b", text):
            return horizon
    return "8h"


def _symbol(text: str) -> str:
    has_btc = "BTCUSDT" in text
    has_eth = "ETHUSDT" in text
    if has_eth and not has_btc:
        return "ETHUSDT"
    return "BTCUSDT"


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 3}


def _too_close(text: str, known: list[str]) -> bool:
    tokens = _tokens(text)
    if len(tokens) < 4:
        return True
    for other in known:
        both = tokens | _tokens(other)
        if not both:
            continue
        if len(tokens & _tokens(other)) / len(both) >= 0.55:
            return True
    return False


def _request_data(stores: Stores, evidence: EvidenceItem, column: str, now: datetime) -> None:
    request_id = "dar_" + sha256_hex(f"{evidence.id}|{column}")[:26].upper()
    if stores.documents.get(DatasetAcquisitionRequest, request_id) is not None:
        return
    stores.documents.put(
        DatasetAcquisitionRequest(
            id=request_id,
            requested_by=AgentRole.DATA_ENGINEERING,
            description=(
                f"Evidence {evidence.id} requires {column}, which is not in the "
                "research panel. No hypothesis was registered."
            ),
            columns=(column,),
            rationale=evidence.claim[:500],
            source_draft_title=evidence.id,
            created_at=now,
        ),
        recorded_by=AgentRole.DATA_ENGINEERING.value,
    )
    stores.audit.record(
        AgentRole.DATA_ENGINEERING.value,
        "data.requested",
        request_id,
        {"column": column, "evidence_id": evidence.id},
    )
