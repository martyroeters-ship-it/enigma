"""Strategy interface v1 (QUANTITATIVE_METHODOLOGY §6) for bar strategies.

This module is copied verbatim into every sandbox, so it may import only the standard library
and NumPy.

Contract:
- `configure(params)` is called once on a fresh instance before any other method.
- `target_positions(data)` receives the columns listed in `required_columns()` for ONE symbol,
  ordered by bar, as 1-D arrays of equal length n. Row t is known at `available_at[t]` (bar
  close). It must return n target positions in [-1, 1], where element t is decided using only
  rows 0..t. Validation checks this by re-running on truncated inputs (any difference is
  look-ahead), so implementations must also accept short inputs (return 0 during warm-up).
- Datetime columns (`ts`, `available_at`, `funding_time`) arrive as int64 epoch nanoseconds;
  missing values are `NAT_INT64`. Missing floats are NaN.
- No I/O, network, wall-clock reads, or randomness without a seed taken from `params`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

INTERFACE_VERSION = "1.0.0"
NAT_INT64 = int(np.iinfo(np.int64).min)

ParamValue = float | int | str


@dataclass(frozen=True)
class StrategyMeta:
    name: str
    version: str
    hypothesis_id: str = ""
    cost_model_version: str = ""


class Strategy:
    """Base class for bar strategies. Subclasses override every method below."""

    def configure(self, params: Mapping[str, ParamValue]) -> None:
        self.params = dict(params)

    def required_columns(self) -> tuple[str, ...]:
        raise NotImplementedError

    def metadata(self) -> StrategyMeta:
        raise NotImplementedError

    def target_positions(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError
