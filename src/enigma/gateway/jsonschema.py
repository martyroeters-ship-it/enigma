"""Minimal, deterministic JSON Schema utilities (no third-party dependency).

Supports the subset Pydantic emits and structured-output APIs accept: type, properties,
required, additionalProperties, items, enum, const, anyOf/oneOf/allOf, $ref/$defs,
min/max length/items, minimum/maximum (incl. exclusive), pattern.
"""

from __future__ import annotations

import copy
import re
from typing import Any

_TYPE_CHECKS: dict[str, Any] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}

_MAX_REF_DEPTH = 32


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"only local $ref supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    if not isinstance(node, dict):
        raise ValueError(f"$ref {ref!r} does not resolve to a schema")
    return node


def validate(instance: Any, schema: dict[str, Any]) -> list[str]:
    """Return a list of human-readable violations (empty if valid)."""
    errors: list[str] = []
    _validate(instance, schema, schema, "$", errors, 0)
    return errors


def _validate(  # noqa: PLR0912, PLR0915 - flat keyword dispatch
    inst: Any, schema: dict[str, Any], root: dict[str, Any], path: str, errs: list[str], depth: int
) -> None:
    if depth > _MAX_REF_DEPTH:
        errs.append(f"{path}: schema nesting too deep")
        return
    if "$ref" in schema:
        _validate(inst, _resolve_ref(schema["$ref"], root), root, path, errs, depth + 1)
    if "allOf" in schema:
        for sub in schema["allOf"]:
            _validate(inst, sub, root, path, errs, depth + 1)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            branch_errors = []
            for sub in schema[key]:
                e: list[str] = []
                _validate(inst, sub, root, path, e, depth + 1)
                branch_errors.append(e)
            passing = sum(1 for e in branch_errors if not e)
            if passing == 0 or (key == "oneOf" and passing > 1):
                errs.append(f"{path}: does not match {key}")
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPE_CHECKS[t](inst) for t in types if t in _TYPE_CHECKS):
            errs.append(f"{path}: expected {'|'.join(types)}, got {type(inst).__name__}")
            return
    if "enum" in schema and inst not in schema["enum"]:
        errs.append(f"{path}: {inst!r} not in enum {schema['enum']}")
    if "const" in schema and inst != schema["const"]:
        errs.append(f"{path}: expected const {schema['const']!r}")
    if isinstance(inst, str):
        if "minLength" in schema and len(inst) < schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(inst) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], inst):
            errs.append(f"{path}: does not match pattern {schema['pattern']!r}")
    if isinstance(inst, int | float) and not isinstance(inst, bool):
        if "minimum" in schema and inst < schema["minimum"]:
            errs.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and inst > schema["maximum"]:
            errs.append(f"{path}: above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and inst <= schema["exclusiveMinimum"]:
            errs.append(f"{path}: not above {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and inst >= schema["exclusiveMaximum"]:
            errs.append(f"{path}: not below {schema['exclusiveMaximum']}")
    if isinstance(inst, list):
        if "minItems" in schema and len(inst) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(inst) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(inst):
                _validate(item, schema["items"], root, f"{path}[{i}]", errs, depth + 1)
    if isinstance(inst, dict):
        props: dict[str, Any] = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in inst:
                errs.append(f"{path}: missing required property {name!r}")
        for name, value in inst.items():
            if name in props:
                _validate(value, props[name], root, f"{path}.{name}", errs, depth + 1)
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected property {name!r}")
            elif isinstance(schema.get("additionalProperties"), dict):
                _validate(
                    value, schema["additionalProperties"], root, f"{path}.{name}", errs, depth + 1
                )


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with every local $ref expanded and $defs removed."""

    def walk(node: Any, depth: int) -> Any:
        if depth > _MAX_REF_DEPTH:
            raise ValueError("recursive schema cannot be inlined")
        if isinstance(node, dict):
            if "$ref" in node:
                target = walk(_resolve_ref(node["$ref"], schema), depth + 1)
                rest = {k: walk(v, depth + 1) for k, v in node.items() if k != "$ref"}
                return {**target, **rest}
            return {
                k: walk(v, depth + 1) for k, v in node.items() if k not in ("$defs", "definitions")
            }
        if isinstance(node, list):
            return [walk(v, depth + 1) for v in node]
        return node

    result = walk(copy.deepcopy(schema), 0)
    assert isinstance(result, dict)
    return result


_GEMINI_KEYS = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "properties",
    "required",
    "items",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "anyOf",
    "propertyOrdering",
}


def to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON Schema to Gemini's OpenAPI-subset `responseSchema` form."""

    def conv(node: Any) -> Any:  # noqa: PLR0912 - keyword-by-keyword translation
        if not isinstance(node, dict):
            return node
        node = dict(node)
        if "const" in node:
            node["enum"] = [node.pop("const")]
        for key in ("anyOf", "oneOf"):
            if key in node:
                original = node.pop(key)
                branches = [b for b in original if b.get("type") != "null"]
                nullable = len(branches) < len(original)
                if len(branches) == 1:
                    merged = {**branches[0], **{k: v for k, v in node.items() if k != key}}
                    converted: dict[str, Any] = conv(merged)
                    if nullable:
                        converted["nullable"] = True
                    return converted
                node["anyOf"] = branches
        if isinstance(node.get("type"), list):
            types = [t for t in node["type"] if t != "null"]
            if len(types) < len(node["type"]):
                node["nullable"] = True
            node["type"] = types[0] if types else "string"
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k not in _GEMINI_KEYS:
                continue
            if k == "properties":
                out[k] = {name: conv(sub) for name, sub in v.items()}
                out["propertyOrdering"] = list(v.keys())
            elif k == "items":
                out[k] = conv(v)
            elif k == "anyOf":
                out[k] = [conv(b) for b in v]
            elif k == "type" and isinstance(v, str):
                out[k] = v.upper()
            else:
                out[k] = v
        if out.get("type") == "STRING" and "enum" in out:
            out["format"] = "enum"
            out["enum"] = [str(e) for e in out["enum"]]
        return out

    result = conv(inline_refs(schema))
    assert isinstance(result, dict)
    return result


def is_strict_compatible(schema: dict[str, Any]) -> bool:
    """True if OpenAI strict mode would accept the schema unchanged.

    Strict mode requires every object to forbid additional properties and to list every
    property as required.
    """

    def ok(node: Any) -> bool:
        if isinstance(node, list):
            return all(ok(v) for v in node)
        if not isinstance(node, dict):
            return True
        if node.get("type") == "object" or "properties" in node:
            props = node.get("properties", {})
            if node.get("additionalProperties") is not False:
                return False
            if set(node.get("required", [])) != set(props):
                return False
        return all(ok(v) for k, v in node.items() if k not in ("enum", "const", "default"))

    return ok(schema)
