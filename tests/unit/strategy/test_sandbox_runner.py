from __future__ import annotations

import json

import numpy as np
import pytest

from _synthetic import BASIS_SIGN, LEAKY, MOMENTUM, RANDOM_SIGN, make_panel, package
from enigma.schemas.research import StrategyPackage
from enigma.store import Stores
from enigma.strategy import (
    SandboxJob,
    SandboxLimits,
    SandboxRunner,
    SandboxUnavailableError,
    sample_checkpoints,
    symbol_arrays,
    truncation_invariance_check,
)
from enigma.strategy import sandbox as sandbox_module
from enigma.strategy.package import package_content_hash

HEADER = (
    "import numpy as np\nfrom enigma.strategy.interface import Strategy, StrategyMeta\n\n"
    "class S(Strategy):\n"
    "    def required_columns(self):\n        return ('perp_close',)\n"
    "    def metadata(self):\n        return StrategyMeta(name='s', version='1')\n"
    "    def target_positions(self, data):\n"
)


def _raw_package(body: str, columns: tuple[str, ...] = ("perp_close",)) -> StrategyPackage:
    """Package whose source skips the AST pre-check, to exercise the runtime barrier."""
    base = package(MOMENTUM, "Momentum", columns)
    source = HEADER + body
    return base.model_copy(
        update={
            "source_code": source,
            "entrypoint": "S",
            "required_columns": columns,
            "content_hash": package_content_hash(
                source, "S", {}, base.interface_version, base.cost_model, columns, 1
            ),
            "params": {},
        }
    )


@pytest.fixture
def data() -> dict[str, dict[str, np.ndarray]]:
    frame = make_panel(("BTCUSDT",), n_bars=200, seed=3)
    return {"s0": symbol_arrays(frame, "BTCUSDT", ("perp_close", "basis", "ts"))}


def test_sandbox_runs_strategy_and_returns_positions(
    data: dict[str, dict[str, np.ndarray]], stores: Stores
) -> None:
    pkg = package(MOMENTUM, "Momentum", ("perp_close",), {"lookback": 2})
    runner = SandboxRunner(audit=stores.audit)
    result = runner.run(pkg, data, [SandboxJob("j", "s0"), SandboxJob("k", "s0", {"lookback": 5})])
    assert result.ok, result.error
    assert result.isolation == "netns"
    close = data["s0"]["perp_close"]
    expected = np.zeros(close.size)
    expected[2:] = np.sign(close[2:] - close[:-2])
    np.testing.assert_array_equal(result.positions["j"], expected)
    assert not np.array_equal(result.positions["j"], result.positions["k"])
    assert result.meta["required_columns"] == ["perp_close"]
    entries = stores.audit.entries(subject_id=pkg.id)
    assert [e.action for e in entries] == ["sandbox.run"]
    assert entries[0].details["package_hash"] == pkg.content_hash


def test_truncation_invariance_passes_for_pit_and_fails_for_leaky(
    data: dict[str, dict[str, np.ndarray]],
) -> None:
    runner = SandboxRunner()
    points = {"s0": sample_checkpoints(200, 6, seed=1)}
    assert len(points["s0"]) == 7
    for source, entry in ((MOMENTUM, "Momentum"), (RANDOM_SIGN, "RandomSign")):
        ok = truncation_invariance_check(
            package(source, entry, ("perp_close",)), data, points, runner=runner
        )
        assert ok.passed and ok.checks == 7
    leaky = truncation_invariance_check(
        package(LEAKY, "Leaky", ("perp_close",)), data, points, runner=runner
    )
    assert not leaky.passed
    assert {m.checkpoint for m in leaky.mismatches} == set(points["s0"])


@pytest.mark.parametrize(
    ("body", "event"),
    [
        ("        return open('/etc/hostname').read()\n", "open"),
        ("        open('/tmp/enigma-escape.txt', 'w').write('x')\n        return 0\n", "open"),
        ("        import subprocess\n        return 0\n", "import"),
        ("        import os\n        os.system('true')\n        return 0\n", "os.system"),
        ("        import os\n        return os.listdir('/')\n", "os.listdir"),
    ],
)
def test_runtime_policy_denies_and_reports(
    body: str, event: str, data: dict[str, dict[str, np.ndarray]], stores: Stores
) -> None:
    pkg = _raw_package(body)
    result = SandboxRunner(audit=stores.audit).run(
        pkg, data, [SandboxJob("j", "s0")], static_precheck=False
    )
    assert not result.ok
    assert result.positions == {}
    assert any(v["event"] == event for v in result.violations), result.violations
    denied = [e for e in stores.audit.entries(subject_id=pkg.id) if e.action == "sandbox.denied"]
    assert denied and denied[0].details["event"] == event


def test_caught_denial_still_fails_the_run(data: dict[str, dict[str, np.ndarray]]) -> None:
    body = (
        "        try:\n            open('/etc/hostname')\n        except Exception:\n"
        "            pass\n        return np.zeros(data['perp_close'].shape[0])\n"
    )
    result = SandboxRunner().run(
        _raw_package(body), data, [SandboxJob("j", "s0")], static_precheck=False
    )
    assert not result.ok and result.exit_code == 0 and len(result.violations) == 1


def test_resource_limits_and_timeouts(data: dict[str, dict[str, np.ndarray]]) -> None:
    spin = "        while True:\n            pass\n"
    cpu_limited = SandboxRunner(SandboxLimits(cpu_seconds=1, wall_seconds=20)).run(
        _raw_package(spin), data, [SandboxJob("j", "s0")]
    )
    assert not cpu_limited.ok and cpu_limited.exit_code < 0
    assert cpu_limited.cpu_seconds >= 0.9
    wall_limited = SandboxRunner(SandboxLimits(cpu_seconds=30, wall_seconds=1.0)).run(
        _raw_package(spin), data, [SandboxJob("j", "s0")]
    )
    assert not wall_limited.ok and "wall-clock" in (wall_limited.error or "")
    hog = "        return np.ones(4 * 1024**3, dtype=np.uint8)\n"
    mem_limited = SandboxRunner(SandboxLimits(address_space_bytes=1024**3)).run(
        _raw_package(hog), data, [SandboxJob("j", "s0")]
    )
    assert not mem_limited.ok and "MemoryError" in (mem_limited.error or "")


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("        return np.full(data['perp_close'].shape[0], 2.0)\n", "outside"),
        ("        return np.zeros(3)\n", "shape"),
        ("        return np.full(data['perp_close'].shape[0], np.nan)\n", "NaN"),
    ],
)
def test_invalid_positions_rejected(
    body: str, fragment: str, data: dict[str, dict[str, np.ndarray]]
) -> None:
    result = SandboxRunner().run(_raw_package(body), data, [SandboxJob("j", "s0")])
    assert not result.ok and fragment in (result.error or "")


def test_declared_columns_must_match_package(data: dict[str, dict[str, np.ndarray]]) -> None:
    pkg = package(BASIS_SIGN, "BasisSign", ("basis",))
    mismatched = pkg.model_copy(
        update={
            "required_columns": ("basis", "perp_close"),
            "content_hash": package_content_hash(
                pkg.source_code,
                pkg.entrypoint,
                pkg.params,
                pkg.interface_version,
                pkg.cost_model,
                ("basis", "perp_close"),
                1,
            ),
        }
    )
    result = SandboxRunner().run(mismatched, data, [SandboxJob("j", "s0")])
    assert not result.ok and "required_columns" in (result.error or "")


def test_fails_closed_without_isolation(
    monkeypatch: pytest.MonkeyPatch, data: dict[str, dict[str, np.ndarray]], stores: Stores
) -> None:
    monkeypatch.setattr(sandbox_module, "_netns_available", lambda: False)
    monkeypatch.setattr(
        sandbox_module,
        "_docker_backend_status",
        lambda *, refresh=False: (False, "docker disabled for this test"),
    )
    monkeypatch.delenv(sandbox_module.ALLOW_WEAK_ENV, raising=False)
    pkg = package(MOMENTUM, "Momentum", ("perp_close",))
    with pytest.raises(
        SandboxUnavailableError, match="strategy isolation is not available"
    ) as caught:
        SandboxRunner(audit=stores.audit).run(pkg, data, [SandboxJob("j", "s0")])
    assert "ALLOW_WEAK" not in str(caught.value)
    monkeypatch.setenv(sandbox_module.ALLOW_WEAK_ENV, "1")
    with pytest.raises(SandboxUnavailableError):
        SandboxRunner(allow_weak=False, audit=stores.audit).check_isolation("startup")
    weak = SandboxRunner(audit=stores.audit).run(pkg, data, [SandboxJob("j", "s0")])
    assert weak.ok and weak.isolation == "weak"
    actions = [e.action for e in stores.audit.entries(subject_id=pkg.id)]
    assert "sandbox.weak_mode" in actions and "sandbox.denied" in actions


def test_docker_isolation_uses_the_container_and_not_host_numpy(
    monkeypatch: pytest.MonkeyPatch, data: dict[str, dict[str, np.ndarray]]
) -> None:
    seen: dict[str, object] = {}

    def _spawn(box, parent, **kwargs):  # type: ignore[no-untyped-def]
        seen["kwargs"] = kwargs
        (parent / "stdout.log").write_bytes(b"")
        (parent / "stderr.log").write_bytes(b"")
        spec = json.loads((box / "spec.json").read_text(encoding="utf-8"))
        seen["spec"] = spec
        seen["host_numpy"] = (box / "lib" / "numpy").exists()
        out = box / "out"
        out.mkdir()
        (out / "result.json").write_text(
            json.dumps({"ok": False, "error": "stopped in the test double", "cpu_seconds": 0.2}),
            encoding="utf-8",
        )
        return 1, False

    monkeypatch.setattr(sandbox_module, "_netns_available", lambda: False)
    monkeypatch.setattr(
        sandbox_module,
        "_docker_backend_status",
        lambda *, refresh=False: (True, "linux container"),
    )
    monkeypatch.setattr(sandbox_module, "spawn_docker", _spawn)
    pkg = package(MOMENTUM, "Momentum", ("perp_close",))
    result = SandboxRunner().run(pkg, data, [SandboxJob("j", "s0")])
    assert result.isolation == "docker"
    assert result.cpu_seconds == 0.2
    assert seen["host_numpy"] is False
    spec = seen["spec"]
    assert isinstance(spec, dict)
    assert spec["discover_read_roots"] is True
    assert spec["read_roots"] == []
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["cpu_seconds"] == 120
    assert kwargs["memory_bytes"] == 4 * 1024**3


STATEFUL_CHEAT = '''
import numpy as np
from enigma.strategy.interface import Strategy, StrategyMeta

CACHE = {}


class Cheat(Strategy):
    """Remembers the longest series it has seen and answers every call from it (look-ahead)."""

    def required_columns(self):
        return ("perp_close",)

    def metadata(self):
        return StrategyMeta(name="cheat", version="1")

    def target_positions(self, data):
        c = data["perp_close"]
        if c.shape[0] > CACHE.get("n", 0):
            CACHE["n"] = c.shape[0]
            CACHE["c"] = c.copy()
        full = CACHE["c"]
        nxt = np.append(full[1:], full[-1:])
        return np.sign(nxt - full)[: c.shape[0]]
'''


def test_state_kept_between_calls_cannot_hide_look_ahead(
    data: dict[str, dict[str, np.ndarray]],
) -> None:
    points = {"s0": sample_checkpoints(200, 6, seed=2)}
    result = truncation_invariance_check(
        package(STATEFUL_CHEAT, "Cheat", ("perp_close",)), data, points, runner=SandboxRunner()
    )
    assert not result.passed


@pytest.mark.parametrize(
    "body",
    [
        "        return np.zeros(len(open('inputs.npz', 'rb').read()))\n",
        "        return np.zeros(len(open('spec.json').read()))\n",
        "        open('out/s0.npy', 'wb').write(b'x')\n        return 0\n",
    ],
)
def test_strategy_cannot_touch_sandbox_inputs_or_outputs(
    body: str, data: dict[str, dict[str, np.ndarray]]
) -> None:
    result = SandboxRunner().run(
        _raw_package(body), data, [SandboxJob("s0", "s0")], static_precheck=False
    )
    assert not result.ok
    assert result.violations and result.violations[0]["event"] == "open"
