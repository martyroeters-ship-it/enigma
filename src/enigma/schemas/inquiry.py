"""A scientific case is not a strategy template.

The mechanism is free text. The program is a small computation the sandbox can run.
An interpretation written by a model is stored as interpretation and is never evidence.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from enigma.schemas.common import NonEmptyStr, StrictModel, UtcDatetime

ComputationOp = Literal[
    "identity", "zscore", "rank", "product", "difference", "threshold", "regime"
]
Combination = Literal["last", "sum", "mean", "product"]
Sampling = Literal["1h", "8h", "1d"]


class Participant(StrictModel):
    name: NonEmptyStr
    incentive: NonEmptyStr
    constraint: NonEmptyStr


class Prediction(StrictModel):
    statement: NonEmptyStr
    observable: NonEmptyStr
    discriminates: str = ""


class DataNeed(StrictModel):
    observable: NonEmptyStr
    role: Literal["signal", "condition", "outcome"]
    why: NonEmptyStr


class Computation(StrictModel):
    op: ComputationOp
    inputs: tuple[NonEmptyStr, ...] = Field(min_length=1)
    lookback: int = Field(default=21, ge=2)
    level: float | None = None


class ResearchProgram(StrictModel):
    """How the hypothesis will be measured. This is not a catalogue of mechanisms."""

    design_class: NonEmptyStr
    computations: tuple[Computation, ...] = Field(min_length=1)
    combination: Combination = "sum"
    condition: Computation | None = None
    holding_bars: int = Field(default=1, ge=1)
    sampling_frequency: Sampling = "8h"
    symbols: tuple[NonEmptyStr, ...] = Field(default=("BTCUSDT",), min_length=1)


class ScientificCase(StrictModel):
    id: NonEmptyStr
    mechanism: NonEmptyStr
    persistence: NonEmptyStr
    participants: tuple[Participant, ...] = Field(min_length=1)
    structural_conditions: NonEmptyStr
    supporting_evidence_ids: tuple[NonEmptyStr, ...] = Field(min_length=2)
    contradictory_evidence_ids: tuple[str, ...] = ()
    predictions: tuple[Prediction, ...] = Field(min_length=1)
    competing_explanations: tuple[NonEmptyStr, ...] = Field(min_length=1)
    data_needs: tuple[DataNeed, ...] = Field(min_length=1)
    program: ResearchProgram
    interpretation: NonEmptyStr
    model_id: str = ""
    created_at: UtcDatetime


class AcquisitionRecord(StrictModel):
    id: NonEmptyStr
    request_id: NonEmptyStr
    observable: str = ""
    status: Literal["registered", "unmet", "unspecified"]
    provider_id: str = ""
    dataset_version_id: str = ""
    content_hash: str = ""
    assessment: NonEmptyStr
    created_at: UtcDatetime
