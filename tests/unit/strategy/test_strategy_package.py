from __future__ import annotations

import pytest
from _synthetic import BASIS_SIGN, MOMENTUM, cost_model, package

from enigma.core.errors import IntegrityError
from enigma.strategy import (
    INTERFACE_VERSION,
    StrategyRejectedError,
    check_source,
    package_content_hash,
    verify_package,
)

GOOD_HEADER = (
    "import numpy as np\nfrom enigma.strategy.interface import Strategy, StrategyMeta\n\n"
    "class S(Strategy):\n    def target_positions(self, data):\n"
)


def _rules(body: str) -> set[str]:
    return {v.rule for v in check_source(GOOD_HEADER + body, "S")}


def test_good_sources_pass_static_check() -> None:
    assert check_source(BASIS_SIGN, "BasisSign") == ()
    assert check_source(MOMENTUM, "Momentum") == ()


@pytest.mark.parametrize(
    ("body", "rule"),
    [
        ("        import os\n        return 0\n", "import"),
        ("        import socket\n        return 0\n", "import"),
        ("        from importlib import import_module\n        return 0\n", "import"),
        ("        return eval('1')\n", "name"),
        ("        return __import__('os')\n", "name"),
        ("        return open('/etc/passwd')\n", "name"),
        ("        return getattr(np, 'ctypeslib')\n", "name"),
        ("        return ().__class__.__bases__[0].__subclasses__()\n", "dunder"),
        ("        return (x for x in ()).gi_frame.f_back\n", "attribute"),
        ("        return np.load('x.npy')\n", "attribute"),
        ("        return compile('1', 'f', 'eval')\n", "name"),
        ("        return globals()\n", "name"),
    ],
)
def test_static_check_rejects_escape_routes(body: str, rule: str) -> None:
    assert rule in _rules(body)


def test_static_check_rejects_top_level_code_and_missing_entrypoint() -> None:
    rules = {v.rule for v in check_source("import numpy as np\nfor i in range(3):\n    pass\n", "X")}
    assert {"top_level", "entrypoint"} <= rules
    assert {v.rule for v in check_source("def f(:\n", "X")} == {"syntax"}


def test_build_package_hash_is_content_addressed() -> None:
    a = package(BASIS_SIGN, "BasisSign", ("basis",), {"threshold": 0.0}, package_id="pkg_a")
    b = package(BASIS_SIGN, "BasisSign", ("basis",), {"threshold": 0.0}, package_id="pkg_b")
    assert a.content_hash == b.content_hash
    assert a.interface_version == INTERFACE_VERSION
    c = package(BASIS_SIGN, "BasisSign", ("basis",), {"threshold": 0.1})
    d = package(BASIS_SIGN, "BasisSign", ("basis",), {"threshold": 0.0}, cm=cost_model(taker_fee_bps=5))
    assert len({a.content_hash, c.content_hash, d.content_hash}) == 3
    assert a.content_hash == package_content_hash(
        BASIS_SIGN, "BasisSign", {"threshold": 0.0}, INTERFACE_VERSION, cost_model(), ("basis",), 1
    )
    verify_package(a)


def test_verify_package_detects_tampering() -> None:
    a = package(BASIS_SIGN, "BasisSign", ("basis",))
    tampered = a.model_copy(update={"source_code": a.source_code + "\n# changed"})
    with pytest.raises(IntegrityError):
        verify_package(tampered)


def test_build_package_rejects_forbidden_source() -> None:
    with pytest.raises(StrategyRejectedError) as info:
        package(GOOD_HEADER + "        import socket\n        return 0\n", "S", ("perp_close",))
    assert any(v.rule == "import" for v in info.value.violations)
