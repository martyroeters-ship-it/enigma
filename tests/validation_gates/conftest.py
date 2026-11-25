from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest

from _validation_env import Env, EnvFactory, build_env


@pytest.fixture
def make_env(tmp_path: Path) -> Iterator[EnvFactory]:
    opened: list[Env] = []

    def factory(frame: pl.DataFrame, name: str = "env") -> Env:
        env = build_env(tmp_path / name, frame)
        opened.append(env)
        return env

    yield factory
    for env in opened:
        env.service.custody.close()
        env.stores.db.close()
