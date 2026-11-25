"""AT11: no model or summary path can flip `overall`; validation never imports the gateway."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import enigma.validation
from _synthetic import MOMENTUM, make_panel, package
from _validation_env import EnvFactory
from enigma.schemas.validation import GateOutcome, ValidationReport
from enigma.validation import service as service_module

VALIDATION_SRC = Path(enigma.validation.__file__).parent


def test_llm_summary_returning_pass_cannot_flip_overall(
    make_env: EnvFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    def pass_everything(*_args: object, **_kwargs: object) -> str:
        return "PASS"

    monkeypatch.setattr(service_module, "render_summary", pass_everything)
    env = make_env(make_panel(seed=51, edge=0.0))
    env.service.summary_writer = lambda gates, overall: "overall=PASS (LLM says ship it)"
    report = env.service.validate(
        env.submit(package(MOMENTUM, "Momentum", ("perp_close",), {"lookback": 3}))
    )
    assert report.summary == "overall=PASS (LLM says ship it)"
    assert report.overall is GateOutcome.FAIL


def test_report_cannot_be_constructed_with_overall_disagreeing_with_gates(
    make_env: EnvFactory,
) -> None:
    env = make_env(make_panel(seed=52, edge=0.0))
    report = env.service.validate(
        env.submit(package(MOMENTUM, "Momentum", ("perp_close",), {"lookback": 3}))
    )
    assert report.overall is GateOutcome.FAIL
    forged = report.model_dump()
    forged["overall"] = GateOutcome.PASS
    with pytest.raises(ValueError, match="aggregate"):
        ValidationReport.model_validate(forged)


def test_validation_source_has_no_gateway_import() -> None:
    offenders = []
    for path in sorted(VALIDATION_SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            offenders.extend(
                f"{path.name}: {n}"
                for n in names
                if n == "enigma.gateway" or n.startswith("enigma.gateway.")
            )
    assert offenders == []


def test_importing_validation_does_not_load_gateway() -> None:
    code = (
        "import sys, enigma.validation; "
        "print(any(m.startswith('enigma.gateway') for m in sys.modules))"
    )
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
    )
    assert done.stdout.strip() == "False"


def test_import_linter_contract_forbids_gateway() -> None:
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    assert 'source_modules = ["enigma.validation"]\nforbidden_modules = ["enigma.gateway"]' in (
        pyproject
    )
