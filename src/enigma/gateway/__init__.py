"""Provider-independent model gateway.

All model calls in Enigma go through `ModelGateway.complete` (or `StructuredLLM.generate`),
which enforces routing, budgets, the global daily spend ceiling, retries, fallback, caching,
and cost-ledger recording. Unconfigured providers fail closed with `ProviderUnavailableError`.
"""

from enigma.gateway.errors import (
    MissingCredentialsError,
    ProviderCallError,
    RegistryError,
    StructuredOutputError,
)
from enigma.gateway.gateway import (
    DAILY_CAP_ENV,
    CandidateFailure,
    FailureKind,
    GatewayUnavailableError,
    ModelGateway,
    ProviderStatus,
    estimate_input_tokens,
    validate_structured,
)
from enigma.gateway.registry import ModelRegistry, ModelSpec, ProviderSpec, ResolvedRoute
from enigma.gateway.structured import StructuredLLM
from enigma.gateway.types import (
    FinishReason,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    NamedToolChoice,
    ReasoningEffort,
    RequestMetadata,
    ToolCall,
    ToolChoice,
    ToolSpec,
    Usage,
)

__all__ = [
    "DAILY_CAP_ENV",
    "CandidateFailure",
    "FailureKind",
    "FinishReason",
    "GatewayUnavailableError",
    "Message",
    "MessageRole",
    "MissingCredentialsError",
    "ModelGateway",
    "ModelRegistry",
    "ModelRequest",
    "ModelResponse",
    "ModelSpec",
    "NamedToolChoice",
    "ProviderCallError",
    "ProviderSpec",
    "ProviderStatus",
    "ReasoningEffort",
    "RegistryError",
    "RequestMetadata",
    "ResolvedRoute",
    "StructuredLLM",
    "StructuredOutputError",
    "ToolCall",
    "ToolChoice",
    "ToolSpec",
    "Usage",
    "estimate_input_tokens",
    "validate_structured",
]
