"""Provider-neutral request/response contracts for model calls.

These types are the gateway's public interface. Provider adapters translate them to and from
each vendor's wire format; callers never see vendor payloads.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from enigma.core.hashing import sha256_of
from enigma.schemas.common import AgentRole, NonEmptyStr, StrictModel


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ReasoningEffort(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


class ToolSpec(StrictModel):
    """A function the model may call. `parameters` is a JSON Schema object."""

    name: NonEmptyStr = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_\-]{0,63}$")
    description: NonEmptyStr
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})


class ToolCall(StrictModel):
    """A normalized tool invocation requested by the model."""

    id: NonEmptyStr
    name: NonEmptyStr
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str = Field(description="Arguments exactly as the provider returned them")
    arguments_valid: bool = Field(
        description="Arguments parsed as a JSON object and the tool name was offered"
    )
    error: str | None = None


class Message(StrictModel):
    role: MessageRole
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = Field(
        default=(), description="Assistant turns only: tool calls the model made"
    )
    tool_call_id: str | None = Field(default=None, description="Tool turns only")
    name: str | None = Field(default=None, description="Tool turns: name of the tool")

    @model_validator(mode="after")
    def _check_role_fields(self) -> Message:
        if self.role is MessageRole.TOOL and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.tool_calls and self.role is not MessageRole.ASSISTANT:
            raise ValueError("only assistant messages may carry tool_calls")
        return self


class NamedToolChoice(StrictModel):
    name: NonEmptyStr


ToolChoice = Literal["auto", "none", "required"] | NamedToolChoice


class RequestMetadata(StrictModel):
    agent_role: AgentRole | None = None
    task_id: str | None = None
    task_kind: str | None = Field(
        default=None, description="Selects a task-specific route, e.g. intelligence.extract"
    )


class ModelRequest(BaseModel):
    """A provider-neutral completion request.

    `model` is either a registry alias (``fast-structured``), a concrete
    ``provider/model-id`` pin, or ``None`` to let routing choose ordered candidates.
    `response_schema` is a Pydantic model class or a JSON Schema dict.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    messages: tuple[Message, ...] = Field(min_length=1)
    model: str | None = None
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    reasoning_effort: ReasoningEffort | None = None
    response_schema: type[BaseModel] | dict[str, Any] | None = None
    schema_name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_\-]{1,64}$")
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice = "auto"
    metadata: RequestMetadata = Field(default_factory=RequestMetadata)
    cacheable: bool | None = Field(
        default=None,
        description="None: cache only when temperature == 0. True/False: force on/off.",
    )

    @field_validator("response_schema")
    @classmethod
    def _schema_is_object(
        cls, value: type[BaseModel] | dict[str, Any] | None
    ) -> type[BaseModel] | dict[str, Any] | None:
        if isinstance(value, dict) and value.get("type") != "object":
            raise ValueError("response_schema JSON Schema must have type 'object'")
        return value

    @model_validator(mode="after")
    def _check_tools(self) -> ModelRequest:
        names = [t.name for t in self.tools]
        if len(set(names)) != len(names):
            raise ValueError("tool names must be unique")
        if isinstance(self.tool_choice, NamedToolChoice) and self.tool_choice.name not in names:
            raise ValueError(f"tool_choice names unknown tool {self.tool_choice.name!r}")
        if self.tool_choice == "required" and not self.tools:
            raise ValueError("tool_choice 'required' needs at least one tool")
        return self

    def json_schema(self) -> dict[str, Any] | None:
        if self.response_schema is None:
            return None
        if isinstance(self.response_schema, dict):
            return self.response_schema
        return self.response_schema.model_json_schema()

    def resolved_schema_name(self) -> str:
        if self.schema_name:
            return self.schema_name
        if isinstance(self.response_schema, type):
            return self.response_schema.__name__
        return "structured_output"

    @property
    def is_cacheable(self) -> bool:
        if self.cacheable is not None:
            return self.cacheable
        return self.temperature == 0.0

    def canonical(self) -> dict[str, Any]:
        """Hashable, JSON-safe form (schema classes become their JSON Schema)."""
        return {
            "messages": [m.model_dump(mode="json") for m in self.messages],
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "response_schema": self.json_schema(),
            "schema_name": self.resolved_schema_name() if self.response_schema else None,
            "tools": [t.model_dump(mode="json") for t in self.tools],
            "tool_choice": (
                self.tool_choice.model_dump(mode="json")
                if isinstance(self.tool_choice, NamedToolChoice)
                else self.tool_choice
            ),
            "task_kind": self.metadata.task_kind,
        }

    def request_hash(self) -> str:
        return sha256_of(self.canonical())


class Usage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0, description="Includes reasoning tokens")
    reasoning_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelResponse(StrictModel):
    text: str
    parsed: Any = Field(
        default=None, description="Validated structured output (model instance or dict)"
    )
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage
    provider: NonEmptyStr
    alias: str | None = None
    model_id: NonEmptyStr = Field(description="Model id that was requested from the provider")
    model_version: NonEmptyStr = Field(description="Model version reported by the provider")
    latency_ms: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
    cost_is_estimate: bool = Field(
        default=True, description="Prices come from registries/models/pricing.yaml estimates"
    )
    cached: bool = False
    finish_reason: FinishReason
    raw_finish_reason: str | None = None
    raw_response_hash: NonEmptyStr
    request_hash: NonEmptyStr
    reasoning_effort_applied: str | None = Field(
        default=None, description="Provider-native reasoning setting actually sent"
    )
    ignored_parameters: tuple[str, ...] = Field(
        default=(), description="Requested parameters this provider/model could not honour"
    )
    structured_mode: str | None = Field(
        default=None, description="native | tool | json_mode | prompt"
    )
    repaired: bool = False
    attempts: int = Field(default=1, ge=1, description="HTTP attempts incl. retries")
    registry_version: str | None = None
