"""Parse a public RSS document and build a point-in-time count feature."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime

from enigma.core.hashing import sha256_hex, sha256_of
from enigma.schemas.narrative import NarrativeEvent


def parse_rss(xml_text: str) -> tuple[NarrativeEvent, ...]:
    root = ET.fromstring(xml_text)  # noqa: S314
    events: list[NarrativeEvent] = []
    for item in root.iter("item"):
        title = _text(item, "title")
        link = _text(item, "link")
        excerpt = _text(item, "description")
        published = _when(item)
        if not title or not link or published is None:
            continue
        events.append(
            NarrativeEvent(
                id="nre_" + sha256_hex(link)[:26].upper(),
                source_url=link,
                title=title,
                excerpt=excerpt,
                event_time=published,
                available_at=published,
                content_hash=sha256_of({"title": title, "excerpt": excerpt}),
                topics=(),
            )
        )
    events.sort(key=lambda event: (event.available_at, event.id))
    return tuple(events)


def intensity_at(
    events: tuple[NarrativeEvent, ...] | list[NarrativeEvent], bar_time: datetime
) -> int:
    """How many events were public at `bar_time`. Later text does not count."""
    return sum(1 for event in events if event.available_at <= bar_time)


def _text(item: ET.Element, tag: str) -> str:
    node = item.find(tag)
    return "" if node is None or node.text is None else node.text.strip()


def _when(item: ET.Element) -> datetime | None:
    raw = _text(item, "availableAt") or _text(item, "pubDate")
    if not raw:
        return None
    if "T" in raw:
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            return None
        return parsed
    parsed_email = parsedate_to_datetime(raw)
    if parsed_email.tzinfo is None:
        return None
    return parsed_email
