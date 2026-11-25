"""Content-addressed StrategyPackage construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from enigma.core.errors import IntegrityError
from enigma.core.hashing import sha256_of
from enigma.core.ids import new_id
from enigma.schemas.common import Actor
from enigma.schemas.research import CostModel, StrategyPackage
from enigma.strategy.interface import INTERFACE_VERSION, ParamValue
from enigma.strategy.static_check import assert_source_allowed


def package_content_hash(
    source_code: str,
    entrypoint: str,
    params: Mapping[str, ParamValue],
    interface_version: str,
    cost_model: CostModel,
    required_columns: Sequence[str],
    label_horizon_bars: int,
) -> str:
    # StrategyPackage stores NonEmptyStr fields stripped; hash what is stored.
    return sha256_of(
        {
            "source_code": source_code.strip(),
            "entrypoint": entrypoint.strip(),
            "params": dict(params),
            "interface_version": interface_version,
            "cost_model": cost_model.model_dump(mode="json"),
            "required_columns": list(required_columns),
            "label_horizon_bars": label_horizon_bars,
        }
    )


def build_package(
    source_code: str,
    entrypoint: str,
    params: Mapping[str, ParamValue],
    cost_model: CostModel,
    hypothesis_id: str,
    *,
    name: str,
    required_columns: Sequence[str],
    created_by: Actor,
    created_at: datetime,
    label_horizon_bars: int = 1,
    package_id: str | None = None,
    sandbox_profile: str = "default",
) -> StrategyPackage:
    """Statically check `source_code` and wrap it in a content-addressed StrategyPackage.

    Raises `StrategyRejectedError` if the source fails the AST pre-checks.
    """
    assert_source_allowed(source_code, entrypoint)
    columns = tuple(required_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("required_columns must be non-empty and unique")
    content_hash = package_content_hash(
        source_code, entrypoint, params, INTERFACE_VERSION, cost_model, columns, label_horizon_bars
    )
    return StrategyPackage(
        id=package_id or new_id("pkg"),
        hypothesis_id=hypothesis_id,
        name=name,
        source_code=source_code,
        entrypoint=entrypoint,
        params=dict(params),
        required_columns=columns,
        content_hash=content_hash,
        interface_version=INTERFACE_VERSION,
        cost_model=cost_model,
        label_horizon_bars=label_horizon_bars,
        sandbox_profile=sandbox_profile,
        created_by=created_by,
        created_at=created_at,
    )


def verify_package(package: StrategyPackage) -> None:
    """Fail closed if the package content does not match its recorded hash."""
    expected = package_content_hash(
        package.source_code,
        package.entrypoint,
        package.params,
        package.interface_version,
        package.cost_model,
        package.required_columns,
        package.label_horizon_bars,
    )
    if expected != package.content_hash:
        raise IntegrityError(f"package {package.id} content_hash mismatch")
    if package.interface_version != INTERFACE_VERSION:
        raise IntegrityError(
            f"package {package.id} targets interface {package.interface_version},"
            f" runtime is {INTERFACE_VERSION}"
        )
