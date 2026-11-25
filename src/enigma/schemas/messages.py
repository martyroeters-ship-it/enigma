"""Typed message envelope for agent-to-agent communication (ARCHITECTURE.md §5)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from enigma.core.hashing import sha256_of
from enigma.schemas.common import Actor, NonEmptyStr, Sha256, StrictModel, UtcDatetime


class MessageType(StrEnum):
    TASK_ASSIGNED = "TaskAssigned"
    TASK_COMPLETED = "TaskCompleted"
    TASK_FAILED = "TaskFailed"
    SOURCE_INGESTED = "SourceIngested"
    EVIDENCE_EXTRACTED = "EvidenceExtracted"
    DATASET_VERSION_PUBLISHED = "DatasetVersionPublished"
    DATA_INTEGRITY_ALERT = "DataIntegrityAlert"
    HYPOTHESIS_PROPOSED = "HypothesisProposed"
    EXPERIMENT_REGISTERED = "ExperimentRegistered"
    STRATEGY_PACKAGE_READY = "StrategyPackageReady"
    VALIDATION_REQUEST = "ValidationRequest"
    VALIDATION_REPORT = "ValidationReport"
    RISK_VETO = "RiskVeto"
    RISK_CLEARANCE = "RiskClearance"
    KILL_SWITCH_ACTIVATED = "KillSwitchActivated"
    IMPROVEMENT_PROPOSAL = "ImprovementProposal"
    IMPROVEMENT_DECISION = "ImprovementDecision"
    INNOVATION_PROPOSAL = "InnovationProposal"
    REFLECTION_RECORDED = "ReflectionRecorded"
    LESSON_RECORDED = "LessonRecorded"


SCHEMA_VERSION = "1.0.0"


class MessageEnvelope(StrictModel):
    message_id: NonEmptyStr
    correlation_id: NonEmptyStr
    causation_id: str | None = Field(default=None, description="Message that caused this one")
    timestamp_utc: UtcDatetime
    from_agent: Actor
    to_agent: Actor
    message_type: MessageType
    schema_version: NonEmptyStr = SCHEMA_VERSION
    payload: dict[str, Any]
    payload_hash: Sha256

    @model_validator(mode="after")
    def _hash_matches(self) -> MessageEnvelope:
        if sha256_of(self.payload) != self.payload_hash:
            raise ValueError("payload_hash does not match payload")
        return self
