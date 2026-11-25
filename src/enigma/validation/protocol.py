"""Validation protocol loader. The registry YAML is human-owned and pinned by sha256 here.

Editing `registries/validation/protocol_0.1.0-slice.yaml` without a new protocol version and a
matching pin change in this file makes the validation service refuse to run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from enigma.core.errors import IntegrityError

PROTOCOL_VERSION = "0.1.0-slice"
PROTOCOL_SHA256 = "ccaa3d0f26b8474320b8be8e518c2a3abfed687afb523018782fdbde6f7efc8c"
DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[3] / "registries" / "validation" / "protocol_0.1.0-slice.yaml"
)
REQUIRED_KEYS = ("protocol_version", "required_gates", "annualization_periods_per_year", "gates")


@dataclass(frozen=True)
class ValidationProtocol:
    version: str
    required_gates: tuple[str, ...]
    periods_per_year: float
    gates: dict[str, dict[str, Any]]
    sha256: str

    def gate(self, gate_id: str) -> dict[str, Any]:
        try:
            return self.gates[gate_id]
        except KeyError as exc:
            raise IntegrityError(f"protocol {self.version} has no gate {gate_id}") from exc


def load_protocol(
    path: Path | None = None, expected_sha256: str = PROTOCOL_SHA256
) -> ValidationProtocol:
    """Load and verify the protocol. Fails closed on any hash or structure mismatch."""
    target = path or DEFAULT_PROTOCOL_PATH
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise IntegrityError(f"validation protocol unreadable: {target.name}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise IntegrityError(
            f"validation protocol {target.name} sha256 {digest} does not match pinned"
            f" {expected_sha256}; protocol changes need a new version and pin"
        )
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or any(k not in data for k in REQUIRED_KEYS):
        raise IntegrityError("validation protocol is missing required keys")
    if data["protocol_version"] != PROTOCOL_VERSION:
        raise IntegrityError(
            f"protocol_version {data['protocol_version']!r} != supported {PROTOCOL_VERSION!r}"
        )
    required = tuple(str(g) for g in data["required_gates"])
    gates = {str(k): dict(v) for k, v in data["gates"].items()}
    missing = [g for g in required if g not in gates]
    if not required or missing:
        raise IntegrityError(f"required gates without configuration: {missing}")
    return ValidationProtocol(
        version=str(data["protocol_version"]),
        required_gates=required,
        periods_per_year=float(data["annualization_periods_per_year"]),
        gates=gates,
        sha256=digest,
    )
