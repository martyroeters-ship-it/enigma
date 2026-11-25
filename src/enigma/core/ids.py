"""Prefixed, time-sortable identifiers (`hyp_01J...`), readable in logs and audit trails."""

from __future__ import annotations

import os
import re
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_PREFIX_RE = re.compile(r"^[a-z][a-z0-9]{1,11}$")


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def new_id(prefix: str) -> str:
    """ULID-style id: 48-bit millisecond timestamp + 80 random bits, Crockford base32."""
    if not _PREFIX_RE.match(prefix):
        raise ValueError(f"invalid id prefix {prefix!r}")
    millis = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    return f"{prefix}_{_encode(millis, 10)}{_encode(rand, 16)}"


def id_prefix(identifier: str) -> str:
    prefix, sep, _ = identifier.partition("_")
    if not sep:
        raise ValueError(f"identifier {identifier!r} has no prefix")
    return prefix
