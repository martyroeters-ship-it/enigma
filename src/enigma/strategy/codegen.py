"""LLM strategy codegen with static/sandbox repair loop (no silent template fallback)."""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from enigma.core.ids import new_id
from enigma.gateway.prompts import PromptRegistry, PromptTemplate
from enigma.gateway.protocol import StructuredGenerator, provenance_of
from enigma.schemas.common import AgentRole
from enigma.schemas.governance import Budget, ExecutionMode, ModelProvenance
from enigma.schemas.quant import CodegenAttempt, StrategyGenerationRecord, StrategySpec
from enigma.schemas.research import CostModel, StrategyPackage
from enigma.strategy.package import build_package
from enigma.strategy.sandbox import SandboxJob, SandboxRunner
from enigma.strategy.static_check import check_source

PROMPT_ID = "strategy.codegen"
METHOD = "llm:strategy-codegen-v1"
MAX_ATTEMPTS = 3


def _default_params(spec: StrategySpec) -> dict[str, float | int | str]:
    """Grid defaults plus top-level cadence fields the model often reads as params."""
    params: dict[str, float | int | str] = {p.name: p.values[0] for p in spec.parameters}
    params.setdefault("rebalance_bars", int(spec.rebalance_bars))
    params.setdefault("holding_bars", int(spec.holding_bars))
    params.setdefault("label_horizon_bars", int(spec.label_horizon_bars))
    return params


def _uses_numpy_name(tree: ast.AST) -> bool:
    """True when source references bare `np` (typical after `import numpy as np`)."""
    return any(isinstance(node, ast.Name) and node.id == "np" for node in ast.walk(tree))


def _imports_numpy(tree: ast.AST) -> bool:
    for node in tree.body if isinstance(tree, ast.Module) else ():
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy" or (alias.asname or alias.name) == "np":
                    return True
        elif isinstance(node, ast.ImportFrom) and node.module == "numpy":
            return True
    return False


def _ensure_numpy_import(tree: ast.Module) -> ast.Module:
    """Local models often call np.* without importing numpy — inject the allowlisted import."""
    if not _uses_numpy_name(tree) or _imports_numpy(tree):
        return tree
    import_np = ast.Import(names=[ast.alias(name="numpy", asname="np")])
    # Keep from __future__ first if present; otherwise prepend.
    insert_at = 0
    first = tree.body[0] if tree.body else None
    if isinstance(first, ast.ImportFrom) and first.module == "__future__":
        insert_at = 1
    tree.body.insert(insert_at, import_np)
    return tree


def _normalize_generated_source(source: str) -> str:
    """Mechanical fixes for common LLM slips (not a template strategy body)."""
    # Models often invent self.inputs; the interface passes columns via the data arg.
    source = re.sub(r"\bself\.inputs\b", "data", source)

    class _Rewriter(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            self.generic_visit(node)
            # Drop annotations that reference Mapping/Dict without imports (NameError at runtime).
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            for arg in node.args.kwonlyargs:
                arg.annotation = None
            if node.args.vararg is not None:
                node.args.vararg.annotation = None
            if node.args.kwarg is not None:
                node.args.kwarg.annotation = None
            # Holdout panels can contain NaN funding; wrap returns so positions stay finite.
            if node.name == "target_positions":
                for stmt in node.body:
                    if isinstance(stmt, ast.Return) and stmt.value is not None:
                        stmt.value = ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="np", ctx=ast.Load()),
                                attr="clip",
                            ),
                            args=[
                                ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id="np", ctx=ast.Load()),
                                        attr="nan_to_num",
                                    ),
                                    args=[stmt.value],
                                    keywords=[
                                        ast.keyword(arg="nan", value=ast.Constant(value=0.0))
                                    ],
                                ),
                                ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1.0)),
                                ast.Constant(value=1.0),
                            ],
                            keywords=[],
                        )
            return node

        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name != "StrategyMeta":
                return node
            new_keywords: list[ast.keyword] = []
            for kw in node.keywords:
                if kw.arg == "id":
                    new_keywords.append(ast.keyword(arg="hypothesis_id", value=kw.value))
                else:
                    new_keywords.append(kw)
            node.keywords = new_keywords
            return node

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    assert isinstance(tree, ast.Module)
    tree = _Rewriter().visit(tree)
    assert isinstance(tree, ast.Module)
    tree = _ensure_numpy_import(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return re.sub(r"StrategyMeta\(([^)]*?)\bid\s*=", r"StrategyMeta(\1hypothesis_id=", source)


def _sandbox_smoke(
    package: StrategyPackage,
    required_columns: tuple[str, ...],
    params: dict[str, Any],
) -> tuple[str, ...]:
    """Run a tiny synthetic panel through the sandbox; return error strings or ()."""
    n = 64
    rng = np.random.default_rng(0)
    base = rng.normal(size=n)
    arrays: dict[str, np.ndarray] = {
        col: (base.copy() if "fund" in col or "rate" in col else 100 + np.cumsum(base * 0.01))
        for col in required_columns
    }
    # Seed a few NaNs like real Binance funding panels so missing nan_to_num fails smoke.
    for col, arr in list(arrays.items()):
        if "fund" in col or "rate" in col:
            dirty = np.asarray(arr, dtype=float).copy()
            dirty[0] = np.nan
            dirty[n // 2] = np.nan
            arrays[col] = dirty
    arrays["ts"] = np.arange(n, dtype=np.int64) * 3_600_000_000_000
    data = {
        "s0": {k: np.asarray(v, dtype=np.int64 if k == "ts" else float) for k, v in arrays.items()}
    }
    try:
        result = SandboxRunner().run(package, data, [SandboxJob("smoke", "s0", dict(params))])
    except Exception as exc:
        return (str(exc),)
    if not result.ok:
        return (result.error or "sandbox smoke failed",)
    positions = result.positions.get("smoke")
    if positions is None or len(positions) != n:
        return ("sandbox smoke returned wrong position length",)
    if not np.isfinite(positions).all():
        return ("sandbox smoke produced non-finite positions",)
    if float(np.nanmax(np.abs(positions))) > 1.0 + 1e-9:
        return ("sandbox smoke positions outside [-1, 1]",)
    return ()


class GeneratedStrategySource(BaseModel):
    """Structured LLM output for one codegen attempt."""

    entrypoint: str = Field(min_length=1)
    source_code: str = Field(min_length=1)
    notes: str = ""


class StrategyCodeGenerator:
    """Generate a StrategyPackage from a StrategySpec via registered prompts.

    Failures are explicit (`ok=False` + `failure_reason`). Never substitutes canned source.
    """

    def __init__(
        self,
        llm: StructuredGenerator,
        prompts: PromptRegistry,
        *,
        prompt_id: str = PROMPT_ID,
        model: str | None = None,
        max_output_tokens: int = 2500,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.prompt_id = prompt_id
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.max_attempts = max_attempts
        self.last_package: StrategyPackage | None = None

    def generate(
        self,
        spec: StrategySpec,
        *,
        cost_model: CostModel,
        actor: AgentRole = AgentRole.STRATEGY_ENGINEERING,
        task_id: str | None,
        budget: Budget,
        now: datetime,
        role: Literal["candidate", "baseline"] = "candidate",
        params: dict[str, float | int | str | bool] | None = None,
    ) -> StrategyGenerationRecord:
        prompt = self.prompts.get(self.prompt_id)
        attempts: list[CodegenAttempt] = []
        errors: list[str] = []
        package: StrategyPackage | None = None
        resolved_params: dict[str, float | int | str | bool] = (
            dict(params) if params else dict(_default_params(spec))
        )

        for index in range(1, self.max_attempts + 1):
            try:
                source, prov = self._call(prompt, spec, actor, task_id, budget, errors)
            except Exception as exc:
                attempts.append(
                    CodegenAttempt(
                        index=index,
                        entrypoint="",
                        source_hash="",
                        stage_reached="llm",
                        errors=(str(exc),),
                        provenance=None,
                    )
                )
                errors = [str(exc)]
                continue

            normalized = _normalize_generated_source(source.source_code)
            violations = check_source(normalized, source.entrypoint)
            if violations:
                detail = tuple(v.describe() for v in violations)
                attempts.append(
                    CodegenAttempt(
                        index=index,
                        entrypoint=source.entrypoint,
                        source_hash="",
                        stage_reached="static",
                        errors=detail,
                        provenance=prov,
                    )
                )
                errors = list(detail)
                continue

            try:
                package = build_package(
                    normalized,
                    source.entrypoint,
                    resolved_params,
                    cost_model,
                    spec.hypothesis_id,
                    name=spec.name,
                    required_columns=spec.required_columns,
                    created_by=actor,
                    created_at=now,
                    label_horizon_bars=spec.label_horizon_bars,
                )
            except Exception as exc:
                attempts.append(
                    CodegenAttempt(
                        index=index,
                        entrypoint=source.entrypoint,
                        source_hash="",
                        stage_reached="static",
                        errors=(str(exc),),
                        provenance=prov,
                    )
                )
                errors = [str(exc)]
                continue

            smoke_errors = _sandbox_smoke(package, spec.required_columns, resolved_params)
            if smoke_errors:
                attempts.append(
                    CodegenAttempt(
                        index=index,
                        entrypoint=source.entrypoint,
                        source_hash=package.content_hash,
                        stage_reached="sandbox",
                        errors=smoke_errors,
                        provenance=prov,
                    )
                )
                errors = list(smoke_errors)
                package = None
                continue

            attempts.append(
                CodegenAttempt(
                    index=index,
                    entrypoint=source.entrypoint,
                    source_hash=package.content_hash,
                    stage_reached="passed",
                    errors=(),
                    provenance=prov,
                )
            )
            break

        self.last_package = package
        ok = package is not None
        return StrategyGenerationRecord(
            id=new_id("sgrec"),
            spec_id=spec.id,
            hypothesis_id=spec.hypothesis_id,
            role=role,
            execution_mode=ExecutionMode.LLM,
            prompt_ref=prompt.ref,
            attempts=tuple(attempts),
            ok=ok,
            package_id=package.id if package else None,
            package_hash=package.content_hash if package else None,
            checks={"static": ok, "sandbox_smoke": ok},
            failure_reason=None if ok else "codegen_failed_after_repairs",
            created_by=actor,
            created_at=now,
        )

    def _call(
        self,
        prompt: PromptTemplate,
        spec: StrategySpec,
        actor: AgentRole,
        task_id: str | None,
        budget: Budget,
        errors: list[str],
    ) -> tuple[GeneratedStrategySource, ModelProvenance]:
        param_keys = ", ".join(p.name for p in spec.parameters) or "(none)"
        allowed_params = ", ".join(
            sorted(
                {
                    *(p.name for p in spec.parameters),
                    "rebalance_bars",
                    "holding_bars",
                    "label_horizon_bars",
                }
            )
        )
        system, user = prompt.render(
            spec_json=json.dumps(spec.model_dump(mode="json"), indent=2)[:8000],
            param_keys=param_keys,
            allowed_params=allowed_params,
            required_columns=", ".join(spec.required_columns),
            rebalance_bars=str(spec.rebalance_bars),
            holding_bars=str(spec.holding_bars),
            prior_errors="; ".join(errors) if errors else "(none)",
        )
        traced = self.llm.generate_traced(
            GeneratedStrategySource,
            system,
            user,
            actor,
            task_id,
            budget,
            task_kind="strategy.codegen",
            model=self.model,
            max_output_tokens=self.max_output_tokens,
        )
        return traced.value, provenance_of(traced.response, prompt.ref)
