"""Offline harness for `enigma.data`.

`tests/fixtures/data/` holds RECORDED REAL responses, downloaded once on 2026-11-25 from
https://data.binance.vision/data/... (zips + .CHECKSUM, stored under the same relative paths)
and https://www.okx.com/api/v5/public/funding-rate-history. They are served through an
`httpx.MockTransport`; nothing here is live operation. The only constructed responses are
HTTP 404 for unrecorded paths and OKX's empty end-of-history page.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from enigma.data.http import PoliteHttp

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "data"
BINANCE_FIXTURES = FIXTURES / "binance"
OKX_FIXTURES = FIXTURES / "okx"
NOW = datetime(2026, 11, 25, 10, 0, tzinfo=UTC)


@dataclass
class RecordedServer:
    """Serves recorded files by URL path; 404 for anything not recorded."""

    root: Path = BINANCE_FIXTURES
    seen: list[str] = field(default_factory=list)
    user_agents: list[str] = field(default_factory=list)
    overrides: dict[str, Callable[[httpx.Request], httpx.Response]] = field(default_factory=dict)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.seen.append(url)
        self.user_agents.append(request.headers.get("User-Agent", ""))
        if url in self.overrides:
            return self.overrides[url](request)
        if request.url.host == "www.okx.com":
            after = request.url.params.get("after")
            name = "btc-page1.json" if after is None else "btc-page2.json"
            path = OKX_FIXTURES / name
            if after is not None and after != _okx_page1_oldest():
                # Same body OKX returned live on 2026-11-25 for a range with no more history.
                return httpx.Response(200, json={"code": "0", "data": [], "msg": ""})
            return httpx.Response(200, content=path.read_bytes())
        relative = request.url.path.removeprefix("/data/")
        path = self.root / relative
        if not path.is_file():
            return httpx.Response(404)
        return httpx.Response(200, content=path.read_bytes())


def _okx_page1_oldest() -> str:
    data = json.loads((OKX_FIXTURES / "btc-page1.json").read_text())["data"]
    return str(min(int(d["fundingTime"]) for d in data))


def make_http(server: RecordedServer) -> PoliteHttp:
    return PoliteHttp(
        transport=httpx.MockTransport(server), min_interval_s=0.0, sleep=lambda _s: None
    )


def fixture_bytes(relative: str) -> bytes:
    return (BINANCE_FIXTURES / relative).read_bytes()
