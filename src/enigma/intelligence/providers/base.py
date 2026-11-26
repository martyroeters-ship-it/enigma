"""Search provider protocol and shared helpers."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from enigma.core.clock import Clock
from enigma.core.errors import ProviderUnavailableError
from enigma.core.hashing import canonical_json
from enigma.intelligence.http import HttpClient, HttpResponse
from enigma.intelligence.models import SearchHit
from enigma.intelligence.untrusted import clean_text

DEFAULT_MAILTO = "enigma-research@users.noreply.github.com"

_TAG_RE = re.compile(r"<[^>]+>")


@runtime_checkable
class SearchProvider(Protocol):
    name: str
    multilingual: bool

    def search(self, query: str, limit: int, language: str | None) -> list[SearchHit]: ...


def require_env(var: str, provider: str) -> str:
    """Read a credential at call time; fail closed when it is not configured."""
    value = os.environ.get(var, "").strip()
    if not value:
        raise ProviderUnavailableError(f"{provider} requires {var}; it is not configured")
    return value


def mailto() -> str:
    return os.environ.get("ENIGMA_CONTACT_EMAIL", DEFAULT_MAILTO)


def strip_markup(text: str | None) -> str | None:
    """Remove tags (e.g. Crossref JATS) and clean; returns None for empty text."""
    if not text:
        return None
    cleaned = clean_text(_TAG_RE.sub(" ", text))
    return cleaned or None


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


class ProviderBase:
    """Stores per-record raw bytes next to the full response, both content-addressed."""

    name: str = "base"
    multilingual: bool = False

    def __init__(self, client: HttpClient, clock: Clock) -> None:
        self.client = client
        self.clock = clock

    def _record_json(self, record: Any) -> tuple[str, str]:
        return self.client.raw_store.put(canonical_json(record).encode("utf-8"), ".json")

    def _record_bytes(self, data: bytes, suffix: str) -> tuple[str, str]:
        return self.client.raw_store.put(data, suffix)

    def _provenance(
        self, response: HttpResponse, record_uri: str, record_hash: str, query: str, rank: int
    ) -> dict[str, Any]:
        return {
            "provider": self.name,
            "query": query,
            "rank": rank,
            "raw_uri": record_uri,
            "content_hash": record_hash,
            "response_uri": response.raw_uri,
            "response_hash": response.content_hash,
            "retrieved_at": self.clock.now(),
        }


__all__ = [
    "ProviderBase",
    "SearchHit",
    "SearchProvider",
    "mailto",
    "parse_date",
    "require_env",
    "strip_markup",
]
