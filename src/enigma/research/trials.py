"""Pre-registered trial plans. Every trial in the plan is registered before it runs (G06)."""

from __future__ import annotations

from itertools import product

from enigma.schemas.research import Hypothesis


def trial_plan(hypothesis: Hypothesis) -> list[dict[str, float | int | str]]:
    """Grid over the hypothesis search space in deterministic order, truncated to the budget."""
    keys = sorted(hypothesis.search_space)
    grid = [
        dict(zip(keys, values, strict=True))
        for values in product(*(hypothesis.search_space[k] for k in keys))
    ]
    return grid[: hypothesis.trial_budget] if grid else [dict(hypothesis.parameters)]
