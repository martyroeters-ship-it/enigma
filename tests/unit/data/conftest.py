from __future__ import annotations

from pathlib import Path

import pytest
from data_harness import NOW, RecordedServer, make_http

from enigma.core.clock import FrozenClock
from enigma.data.binance import BinanceArchiveClient


@pytest.fixture
def server() -> RecordedServer:
    return RecordedServer()


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(NOW)


@pytest.fixture
def client(tmp_path: Path, server: RecordedServer, clock: FrozenClock) -> BinanceArchiveClient:
    return BinanceArchiveClient(tmp_path / "raw", http=make_http(server), clock=clock)
