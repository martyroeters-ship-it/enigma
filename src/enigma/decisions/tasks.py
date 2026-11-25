"""Frozen decision items. The set is generated, then hashed, then scored."""

from __future__ import annotations

from dataclasses import dataclass

from enigma.core.hashing import sha256_of

CATEGORIES = ("classification", "anomaly", "relevance", "ranking", "routing", "narrative")
LABELS = ("yes", "no")
SET_SIZE = 216


@dataclass(frozen=True)
class DecisionItem:
    id: str
    category: str
    text: str
    label: str


def frozen_items(n: int = SET_SIZE) -> tuple[DecisionItem, ...]:
    if n < 200:
        raise ValueError("the frozen set must contain at least 200 items")
    items = []
    for index in range(n):
        category = CATEGORIES[index % len(CATEGORIES)]
        label = LABELS[index % len(LABELS)]
        items.append(
            DecisionItem(
                id=f"dec_{index:04d}",
                category=category,
                text=f"{label} {category} item {index}",
                label=label,
            )
        )
    return tuple(items)


def label_set_hash(items: tuple[DecisionItem, ...]) -> str:
    return sha256_of(
        [
            {"id": item.id, "category": item.category, "text": item.text, "label": item.label}
            for item in items
        ]
    )
