from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from enigma.store import Stores


@pytest.fixture
def stores(tmp_path: Path) -> Iterator[Stores]:
    s = Stores.open(tmp_path / "control.sqlite3")
    yield s
    s.db.close()
