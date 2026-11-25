"""AST pre-checks on strategy source. A first, cheap barrier; the sandbox is the real boundary."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from enigma.core.errors import ConstitutionViolationError

ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "numpy",
        "math",
        "statistics",
        "dataclasses",
        "typing",
        "enigma.strategy.interface",
    }
)

FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "__import__",
        "__builtins__",
        "__loader__",
        "__spec__",
        "eval",
        "exec",
        "open",
        "compile",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "input",
        "help",
        "exit",
        "quit",
    }
)

# Frame / code / generator internals are the classic routes out of a restricted namespace.
FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "gi_frame",
        "gi_code",
        "cr_frame",
        "cr_code",
        "ag_frame",
        "ag_code",
        "f_globals",
        "f_locals",
        "f_back",
        "f_builtins",
        "f_code",
        "tb_frame",
        "tb_next",
        "co_code",
        "func_globals",
        "mro",
        # NumPy file / process / FFI surfaces.
        "ctypeslib",
        "f2py",
        "testing",
        "distutils",
        "load",
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "loadtxt",
        "genfromtxt",
        "fromfile",
        "tofile",
        "memmap",
        "fromregex",
        "DataSource",
        "show_config",
    }
)

ALLOWED_DUNDER_ATTRIBUTES: frozenset[str] = frozenset({"__init__"})

_ALLOWED_TOP_LEVEL = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.FunctionDef,
    ast.Assign,
    ast.AnnAssign,
)


class StrategyRejectedError(ConstitutionViolationError):
    """Strategy source failed static pre-checks; it is never executed."""

    def __init__(self, violations: tuple[StaticViolation, ...]) -> None:
        self.violations = violations
        super().__init__("; ".join(v.describe() for v in violations))


@dataclass(frozen=True)
class StaticViolation:
    line: int
    rule: str
    detail: str

    def describe(self) -> str:
        return f"line {self.line}: {self.rule}: {self.detail}"


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _import_allowed(module: str) -> bool:
    return module in ALLOWED_IMPORTS


class _Checker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[StaticViolation] = []

    def _add(self, node: ast.AST, rule: str, detail: str) -> None:
        self.violations.append(StaticViolation(getattr(node, "lineno", 0), rule, detail))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _import_allowed(alias.name):
                self._add(node, "import", f"module {alias.name!r} is not allowlisted")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level:
            self._add(node, "import", "relative imports are not allowed")
        elif not _import_allowed(module):
            self._add(node, "import", f"module {module!r} is not allowlisted")
        for alias in node.names:
            if alias.name == "*":
                self._add(node, "import", "star imports are not allowed")
            elif _is_dunder(alias.name) and module != "__future__":
                self._add(node, "dunder", f"importing {alias.name!r}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self._add(node, "name", f"use of {node.id!r} is forbidden")
        elif _is_dunder(node.id):
            self._add(node, "dunder", f"use of {node.id!r} is forbidden")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if _is_dunder(attr) and attr not in ALLOWED_DUNDER_ATTRIBUTES:
            self._add(node, "dunder", f"attribute access {attr!r} is forbidden")
        elif attr in FORBIDDEN_ATTRIBUTES:
            self._add(node, "attribute", f"attribute access {attr!r} is forbidden")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self._add(node, "scope", "global statements are not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._add(node, "scope", "nonlocal statements are not allowed")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add(node, "async", "async code is not allowed")

    def visit_Await(self, node: ast.Await) -> None:
        self._add(node, "async", "async code is not allowed")


def check_source(source: str, entrypoint: str) -> tuple[StaticViolation, ...]:
    """Return every static violation; an empty tuple means the source may enter the sandbox."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        return (StaticViolation(exc.lineno or 0, "syntax", str(exc.msg)),)
    checker = _Checker()
    checker.visit(tree)
    violations = list(checker.violations)
    for index, stmt in enumerate(tree.body):
        is_docstring = (
            index == 0
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
        if not is_docstring and not isinstance(stmt, _ALLOWED_TOP_LEVEL):
            violations.append(
                StaticViolation(
                    stmt.lineno, "top_level", f"{type(stmt).__name__} not allowed at module level"
                )
            )
    if not entrypoint.isidentifier() or _is_dunder(entrypoint):
        violations.append(StaticViolation(0, "entrypoint", f"invalid entrypoint {entrypoint!r}"))
    else:
        classes = [s for s in tree.body if isinstance(s, ast.ClassDef) and s.name == entrypoint]
        if not classes:
            violations.append(
                StaticViolation(0, "entrypoint", f"no top-level class named {entrypoint!r}")
            )
        else:
            for node in classes[0].body:
                if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                    args = node.args
                    n_required = len(args.args) - len(args.defaults)
                    if n_required > 1:  # beyond self
                        violations.append(
                            StaticViolation(
                                node.lineno,
                                "init_signature",
                                "__init__ may only take self; use configure(params) for parameters",
                            )
                        )
    return tuple(sorted(violations, key=lambda v: (v.line, v.rule, v.detail)))


def assert_source_allowed(source: str, entrypoint: str) -> None:
    violations = check_source(source, entrypoint)
    if violations:
        raise StrategyRejectedError(violations)
