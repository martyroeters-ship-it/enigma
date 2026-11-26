from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from gateway_fixtures import MODELS_DOC, PRICING_DOC, ROUTING_DOC
from pydantic import BaseModel, ValidationError

from enigma.gateway import (
    Message,
    MessageRole,
    ModelRegistry,
    ModelRequest,
    NamedToolChoice,
    RegistryError,
    ToolSpec,
)
from enigma.gateway.gateway import estimate_input_tokens
from enigma.gateway.jsonschema import (
    inline_refs,
    is_strict_compatible,
    to_gemini_schema,
    validate,
)
from enigma.schemas.common import AgentRole

REPO_REGISTRY = Path(__file__).resolve().parents[3] / "registries" / "models"
USER = Message(role=MessageRole.USER, content="hi")


# ------------------------------------------------------------------ shipped registry


def test_shipped_registry_loads_and_is_consistent() -> None:
    reg = ModelRegistry.load(REPO_REGISTRY)
    assert reg.version.startswith("sha256:")
    assert reg.pricing.estimate is True
    for spec in reg.models.values():
        provider = reg.providers[spec.provider]
        if not provider.local:
            assert reg.price_for(spec) is not None, f"{spec.alias} has no pricing entry"
    for role in AgentRole:
        route = reg.route(role, None)
        assert route.candidates
        local_last = reg.providers[reg.models[route.candidates[-1]].provider].local
        assert local_last, f"{role}: last candidate should be a local model"
    for spec in reg.models.values():
        for env in reg.providers[spec.provider].api_key_env:
            assert env.isupper()


def test_version_hash_is_deterministic_and_content_sensitive() -> None:
    a = ModelRegistry.from_documents(MODELS_DOC, PRICING_DOC, ROUTING_DOC)
    b = ModelRegistry.from_documents(copy.deepcopy(MODELS_DOC), PRICING_DOC, ROUTING_DOC)
    pricing = copy.deepcopy(PRICING_DOC)
    pricing["prices"]["openai/gpt-plain"]["input"] = 2.5
    c = ModelRegistry.from_documents(MODELS_DOC, pricing, ROUTING_DOC)
    assert a.version == b.version != c.version


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m, p, r: r["defaults"].update(candidates=["nope"]),
        lambda m, p, r: m["models"]["o-plain"].update(provider="ghost"),
        lambda m, p, r: p["prices"].update({"ghost/x": {"input": 1, "output": 1, "source": "s"}}),
        lambda m, p, r: r["roles"].update({"not_a_role": {}}),
        lambda m, p, r: r["defaults"].pop("max_cost_per_call_usd"),
        lambda m, p, r: m["providers"]["openai"].pop("api_key_env"),
        lambda m, p, r: m["models"]["o-reason"]["capabilities"].pop("reasoning_efforts"),
    ],
)
def test_invalid_registry_rejected(mutate: Any) -> None:
    m, p, r = copy.deepcopy(MODELS_DOC), copy.deepcopy(PRICING_DOC), copy.deepcopy(ROUTING_DOC)
    mutate(m, p, r)
    with pytest.raises(RegistryError):
        ModelRegistry.from_documents(m, p, r)


def test_missing_registry_file(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="missing"):
        ModelRegistry.load(tmp_path)


def test_resolve_alias_concrete_and_unknown() -> None:
    reg = ModelRegistry.from_documents(MODELS_DOC, PRICING_DOC, ROUTING_DOC)
    assert reg.resolve_model("o-plain").model_id == "gpt-plain"
    assert reg.resolve_model("openai/gpt-plain").alias == "o-plain"
    pin = reg.resolve_model("openai/gpt-new")
    assert pin.model_id == "gpt-new" and not pin.capabilities.tools
    assert reg.price_for(pin) is None
    with pytest.raises(RegistryError):
        reg.resolve_model("nobody/x")


# ------------------------------------------------------------------ request types


def test_request_validation() -> None:
    with pytest.raises(ValidationError):
        Message(role=MessageRole.TOOL, content="x")
    with pytest.raises(ValidationError):
        ModelRequest(messages=(USER,), tool_choice=NamedToolChoice(name="missing"))
    with pytest.raises(ValidationError):
        ModelRequest(messages=(USER,), tool_choice="required")
    with pytest.raises(ValidationError):
        ModelRequest(messages=(USER,), response_schema={"type": "array"})
    tool = ToolSpec(name="t", description="d")
    with pytest.raises(ValidationError):
        ModelRequest(messages=(USER,), tools=(tool, tool))
    with pytest.raises(ValidationError):
        ModelRequest(messages=())


def test_request_hash_is_canonical() -> None:
    class Out(BaseModel):
        x: int

    a = ModelRequest(messages=(USER,), response_schema=Out)
    b = ModelRequest(messages=(USER,), response_schema=Out.model_json_schema(), schema_name="Out")
    assert a.request_hash() == b.request_hash()
    assert a.request_hash() != ModelRequest(messages=(USER,)).request_hash()
    assert a.is_cacheable
    assert not ModelRequest(messages=(USER,), temperature=None).is_cacheable


def test_token_estimate_is_conservative() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 100  # ~1000 real tokens
    est = estimate_input_tokens(
        ModelRequest(messages=(Message(role=MessageRole.USER, content=text),))
    )
    assert est >= 1500
    cjk = "比特币资金费率" * 100
    est_cjk = estimate_input_tokens(
        ModelRequest(messages=(Message(role=MessageRole.USER, content=cjk),))
    )
    assert est_cjk >= 700


# ------------------------------------------------------------------ json schema helpers


SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["a", "b"],
    "properties": {
        "a": {"$ref": "#/$defs/Pos"},
        "b": {"anyOf": [{"type": "string", "enum": ["x", "y"]}, {"type": "null"}]},
        "c": {"type": "array", "items": {"type": "integer"}, "maxItems": 2},
    },
    "$defs": {"Pos": {"type": "integer", "minimum": 1}},
}


def test_validate_subset() -> None:
    assert validate({"a": 1, "b": None}, SCHEMA) == []
    assert validate({"a": 1, "b": "x", "c": [1, 2]}, SCHEMA) == []
    errors = validate({"a": 0, "b": "z", "c": [1, 2, "3"], "d": 1}, SCHEMA)
    joined = " | ".join(errors)
    assert "below minimum" in joined
    assert "anyOf" in joined
    assert "more than 2 items" in joined
    assert "$.c[2]" in joined
    assert "unexpected property 'd'" in joined
    assert validate({"b": None}, SCHEMA) == ["$: missing required property 'a'"]
    assert validate({"a": True, "b": None}, SCHEMA)  # bool is not an integer


def test_inline_refs_and_gemini_conversion() -> None:
    inlined = inline_refs(SCHEMA)
    assert "$defs" not in inlined
    assert inlined["properties"]["a"] == {"type": "integer", "minimum": 1}
    gem = to_gemini_schema(SCHEMA)
    assert gem["type"] == "OBJECT"
    assert gem["propertyOrdering"] == ["a", "b", "c"]
    assert gem["properties"]["b"] == {
        "type": "STRING",
        "enum": ["x", "y"],
        "format": "enum",
        "nullable": True,
    }
    assert gem["properties"]["c"]["items"] == {"type": "INTEGER"}
    assert "additionalProperties" not in gem


def test_strict_compatibility() -> None:
    assert not is_strict_compatible(SCHEMA)  # 'c' not required
    strict = copy.deepcopy(SCHEMA)
    strict["required"] = ["a", "b", "c"]
    assert is_strict_compatible(strict)
