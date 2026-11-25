"""Client for Binance public bulk archives on data.binance.vision.

Each archive zip is published with a `.CHECKSUM` file (`<sha256>  <filename>`). Verified zips
are cached content-addressed under `raw_dir/blobs/<aa>/<sha256>.zip`, with a per-URL index
entry under `raw_dir/index/` recording the digest and original retrieval time. A URL whose
index entry points at an existing blob with the recorded digest is never downloaded again.

Monthly archives are used for every month; for months without a monthly archive yet (the
current month, and the previous month shortly after rollover) kline kinds fall back to daily
archives. Funding rates are published monthly only, so the current month has no funding.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl

from enigma.core.clock import Clock, SystemClock
from enigma.core.errors import IntegrityError
from enigma.data.http import PoliteHttp
from enigma.data.normalize import parse_funding_csv, parse_klines_csv, read_zip_csv

BASE_URL = "https://data.binance.vision/data"
KLINE_INTERVALS = ("1h", "2h", "4h", "6h", "8h", "12h", "1d")
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ArchiveKind(StrEnum):
    PERP_KLINES = "perp_klines"
    FUNDING = "funding"
    SPOT_KLINES = "spot_klines"


@dataclass(frozen=True)
class ArchiveFile:
    """One verified archive. `cached` means no network request was made for it this time."""

    kind: ArchiveKind
    symbol: str
    period: str
    url: str
    checksum_url: str
    sha256: str
    path: Path
    retrieved_at: datetime
    cached: bool

    def provenance(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "symbol": self.symbol,
            "period": self.period,
            "url": self.url,
            "checksum_url": self.checksum_url,
            "sha256": self.sha256,
            "retrieved_at": self.retrieved_at.isoformat(),
        }


@dataclass
class FetchResult:
    """Normalized frame plus the archives it came from and the periods that had no archive."""

    frame: pl.DataFrame
    files: list[ArchiveFile] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def archive_url(kind: ArchiveKind, symbol: str, period: str, interval: str = "8h") -> str:
    """URL of a monthly (`YYYY-MM`) or daily (`YYYY-MM-DD`) archive."""
    cadence = "daily" if len(period) == 10 else "monthly"
    if kind is ArchiveKind.FUNDING:
        filename = f"{symbol}-fundingRate-{period}.zip"
        return f"{BASE_URL}/futures/um/{cadence}/fundingRate/{symbol}/{filename}"
    market = "futures/um" if kind is ArchiveKind.PERP_KLINES else "spot"
    filename = f"{symbol}-{interval}-{period}.zip"
    return f"{BASE_URL}/{market}/{cadence}/klines/{symbol}/{interval}/{filename}"


def parse_checksum(text: str) -> str:
    token = text.strip().split()[0].lower() if text.strip() else ""
    if not _SHA_RE.match(token):
        raise IntegrityError(f"malformed CHECKSUM content: {text[:80]!r}")
    return token


def month_starts(start: datetime, end: datetime) -> list[date]:
    """First day of every calendar month overlapping [start, end)."""
    current = date(start.year, start.month, 1)
    months: list[date] = []
    while datetime(current.year, current.month, 1, tzinfo=UTC) < end:
        months.append(current)
        current = date(current.year + (current.month // 12), current.month % 12 + 1, 1)
    return months


class BinanceArchiveClient:
    def __init__(
        self,
        raw_dir: Path,
        http: PoliteHttp | None = None,
        clock: Clock | None = None,
        daily_fallback_months: int = 2,
    ) -> None:
        self.raw_dir = raw_dir
        self.http = http or PoliteHttp()
        self.clock = clock or SystemClock()
        self.daily_fallback_months = daily_fallback_months

    @property
    def http_requests(self) -> int:
        return self.http.requests

    def _index_path(self, url: str) -> Path:
        relative = url.removeprefix(BASE_URL).lstrip("/")
        return self.raw_dir / "index" / f"{relative}.json"

    def _blob_path(self, digest: str) -> Path:
        return self.raw_dir / "blobs" / digest[:2] / f"{digest}.zip"

    def _cached(self, kind: ArchiveKind, symbol: str, period: str, url: str) -> ArchiveFile | None:
        index = self._index_path(url)
        if not index.exists():
            return None
        entry = json.loads(index.read_text(encoding="utf-8"))
        blob = self._blob_path(entry["sha256"])
        if not blob.exists() or hashlib.sha256(blob.read_bytes()).hexdigest() != entry["sha256"]:
            return None
        return ArchiveFile(
            kind=kind,
            symbol=symbol,
            period=period,
            url=url,
            checksum_url=entry["checksum_url"],
            sha256=entry["sha256"],
            path=blob,
            retrieved_at=datetime.fromisoformat(entry["retrieved_at"]),
            cached=True,
        )

    def fetch_archive(
        self, kind: ArchiveKind, symbol: str, period: str, interval: str = "8h"
    ) -> ArchiveFile | None:
        """Download (or reuse) one archive, verifying its sha256. None if not published."""
        if not _SYMBOL_RE.match(symbol):
            raise ValueError(f"invalid symbol {symbol!r}")
        if interval not in KLINE_INTERVALS:
            raise ValueError(f"unsupported interval {interval!r}")
        url = archive_url(kind, symbol, period, interval)
        cached = self._cached(kind, symbol, period, url)
        if cached is not None:
            return cached
        checksum_url = f"{url}.CHECKSUM"
        checksum_response = self.http.get(checksum_url)
        if checksum_response is None:
            return None
        expected = parse_checksum(checksum_response.text)
        response = self.http.get(url)
        if response is None:
            return None
        payload = response.content
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise IntegrityError(f"{url}: sha256 {actual} does not match CHECKSUM {expected}")
        retrieved_at = self.clock.now().astimezone(UTC)
        blob = self._blob_path(actual)
        _atomic_write(blob, payload)
        entry = {
            "url": url,
            "checksum_url": checksum_url,
            "sha256": actual,
            "retrieved_at": retrieved_at.isoformat(),
        }
        _atomic_write(self._index_path(url), json.dumps(entry, sort_keys=True).encode())
        return ArchiveFile(
            kind=kind,
            symbol=symbol,
            period=period,
            url=url,
            checksum_url=checksum_url,
            sha256=actual,
            path=blob,
            retrieved_at=retrieved_at,
            cached=False,
        )

    def _daily_allowed(self, kind: ArchiveKind, month: date) -> bool:
        if kind is ArchiveKind.FUNDING:
            return False
        today = self.clock.now().astimezone(UTC).date()
        months_back = (today.year - month.year) * 12 + (today.month - month.month)
        return 0 <= months_back < self.daily_fallback_months

    def fetch_range(
        self,
        kind: ArchiveKind,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "8h",
    ) -> FetchResult:
        """All archives covering [start, end), parsed and concatenated (not yet trimmed)."""
        result = FetchResult(frame=pl.DataFrame())
        frames: list[pl.DataFrame] = []
        today = self.clock.now().astimezone(UTC).date()
        for month in month_starts(start, end):
            period = month.strftime("%Y-%m")
            file = self.fetch_archive(kind, symbol, period, interval)
            if file is not None:
                result.files.append(file)
                frames.append(_parse(kind, file.path.read_bytes()))
                continue
            if not self._daily_allowed(kind, month):
                result.missing.append(archive_url(kind, symbol, period, interval))
                continue
            day = month
            while day.month == month.month and day < today:
                day_start = datetime(day.year, day.month, day.day, tzinfo=UTC)
                if day_start + timedelta(days=1) > start and day_start < end:
                    daily = self.fetch_archive(kind, symbol, day.isoformat(), interval)
                    if daily is None:
                        result.missing.append(archive_url(kind, symbol, day.isoformat(), interval))
                    else:
                        result.files.append(daily)
                        frames.append(_parse(kind, daily.path.read_bytes()))
                day += timedelta(days=1)
        if frames:
            result.frame = pl.concat(frames, how="vertical")
        else:
            result.frame = _parse(kind, b"")
        return result


def _parse(kind: ArchiveKind, payload: bytes) -> pl.DataFrame:
    csv = read_zip_csv(payload) if payload else b""
    return parse_funding_csv(csv) if kind is ArchiveKind.FUNDING else parse_klines_csv(csv)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)
