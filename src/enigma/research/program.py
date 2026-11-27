"""Compile a research program into strategy source.

The mechanism is not chosen here. The program names computations and columns.
A name that is not a column identifier is rejected. There is no phrase list.
"""

from __future__ import annotations

import re

from enigma.schemas.inquiry import Computation, ResearchProgram

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RICH = frozenset(
    {
        "multivariate",
        "nonlinear",
        "conditional",
        "cross_sectional",
        "probabilistic",
        "regime",
    }
)


def program_problem(program: ResearchProgram) -> str:
    """Empty when the program is internally consistent."""
    computations = list(program.computations)
    if program.condition is not None:
        computations.append(program.condition)
    for comp in computations:
        for name in comp.inputs:
            if _NAME.fullmatch(name) is None:
                return f"column {name!r} is not an identifier"
        if comp.op in {"product", "difference"} and len(comp.inputs) < 2:
            return f"{comp.op} needs two inputs"
        if comp.op in {"threshold", "regime"} and comp.level is None:
            return f"{comp.op} needs a level"
    if program.design_class in _RICH and _is_single_zscore(program):
        return (
            f"design_class {program.design_class!r} is not a single-feature z-score; "
            "the program does not match that claim"
        )
    return ""


def _is_single_zscore(program: ResearchProgram) -> bool:
    if program.condition is not None or len(program.computations) != 1:
        return False
    only = program.computations[0]
    return only.op == "zscore" and len(only.inputs) == 1


def compile_program(program: ResearchProgram, class_name: str = "ResearchProgramStrategy") -> str:
    problem = program_problem(program)
    if problem:
        raise ValueError(problem)
    if _NAME.fullmatch(class_name) is None:
        raise ValueError("class name must be an identifier")
    columns = program_columns(program)
    body = _method_body(program)
    column_tuple = ", ".join(repr(name) for name in columns)
    return (
        "from enigma.strategy.interface import Strategy, StrategyMeta\n"
        "import numpy as np\n\n"
        f"class {class_name}(Strategy):\n"
        "    def required_columns(self):\n"
        f"        return ({column_tuple},)\n\n"
        "    def metadata(self):\n"
        f"        return StrategyMeta({class_name!r}, 'research-program-1')\n\n"
        "    def target_positions(self, data):\n"
        f"{body}"
    )


def program_columns(program: ResearchProgram) -> tuple[str, ...]:
    """Column names the compiled strategy reads. Order is stable."""
    return _columns(program)


def _columns(program: ResearchProgram) -> tuple[str, ...]:
    names: list[str] = []
    comps = list(program.computations)
    if program.condition is not None:
        comps.append(program.condition)
    for comp in comps:
        for name in comp.inputs:
            if name not in names:
                names.append(name)
    return tuple(names)


def _expr(comp: Computation, index: int) -> list[str]:
    look = comp.lookback
    lines = [f"        c{index} = np.zeros(n)"]
    if comp.op == "identity":
        lines.append(f"        c{index}[1:] = data[{comp.inputs[0]!r}][1:]")
    elif comp.op == "zscore":
        lines.extend(_rolling(index, comp.inputs[0], look, "z"))
    elif comp.op == "rank":
        lines.extend(_rolling(index, comp.inputs[0], look, "rank"))
    elif comp.op == "product":
        lines.append(
            "        "
            + f"c{index} = "
            + " * ".join(f"np.nan_to_num(data[{name!r}])" for name in comp.inputs)
        )
    elif comp.op == "difference":
        left, right = comp.inputs[0], comp.inputs[1]
        lines.append(
            f"        c{index} = np.nan_to_num(data[{left!r}]) - np.nan_to_num(data[{right!r}])"
        )
    elif comp.op == "threshold":
        level = float(comp.level or 0.0)
        column = comp.inputs[0]
        lines.append(
            f"        raw = np.nan_to_num(data[{column!r}])\n"
            f"        c{index} = np.where(raw > {level!r}, 1.0, -1.0)"
        )
    elif comp.op == "regime":
        level = float(comp.level or 0.0)
        column = comp.inputs[0]
        lines.append(
            f"        raw = np.nan_to_num(data[{column!r}])\n"
            f"        c{index} = np.where(raw > {level!r}, 1.0, 0.0)"
        )
    lines.append(f"        c{index}[:{look}] = 0.0")
    return lines


def _rolling(index: int, column: str, look: int, kind: str) -> list[str]:
    lines = [
        f"        series = np.nan_to_num(data[{column!r}])",
        f"        for i in range({look} - 1, n):",
        f"            window = series[i - {look} + 1:i + 1]",
    ]
    if kind == "z":
        lines.extend(
            [
                "            mu = window.mean()",
                "            sd = window.std(ddof=1)",
                f"            c{index}[i] = 0.0 if sd <= 1e-12 else (series[i] - mu) / sd",
            ]
        )
    else:
        lines.append(f"            c{index}[i] = float((window <= series[i]).mean()) * 2.0 - 1.0")
    return lines


def _method_body(program: ResearchProgram) -> str:
    lines = [
        "        n = len(next(iter(data.values())))",
        "        if n == 0:",
        "            return np.zeros(0)",
    ]
    for i, comp in enumerate(program.computations):
        lines.extend(_expr(comp, i))
    names = [f"c{i}" for i in range(len(program.computations))]
    if program.combination == "last":
        joined = names[-1]
    elif program.combination == "product":
        joined = " * ".join(names)
    elif program.combination == "mean":
        joined = "(" + " + ".join(names) + f") / {len(names)}"
    else:
        joined = " + ".join(names)
    lines.append(f"        signal = {joined}")
    if program.condition is not None:
        lines.extend(_expr(program.condition, len(program.computations)))
        gate = f"c{len(program.computations)}"
        lines.append(f"        signal = np.where({gate} > 0.0, signal, 0.0)")
    hold = program.holding_bars
    lines.extend(
        [
            "        out = np.zeros(n)",
            "        for i in range(n):",
            f"            out[i] = out[i - 1] if i > 0 and i % {hold} != 0 else signal[i]",
            "        return np.clip(out, -1.0, 1.0)",
        ]
    )
    return "\n".join(lines) + "\n"
