"""Timestamped public narrative. `available_at` is when the text could have been known."""

from __future__ import annotations

from enigma.schemas.common import NonEmptyStr, Sha256, StrictModel, UtcDatetime


class NarrativeEvent(StrictModel):
    id: NonEmptyStr
    source_url: NonEmptyStr
    title: NonEmptyStr
    excerpt: str = ""
    event_time: UtcDatetime
    available_at: UtcDatetime
    content_hash: Sha256
    topics: tuple[str, ...] = ()
