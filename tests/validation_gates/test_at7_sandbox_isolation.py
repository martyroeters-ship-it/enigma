"""AT7: a strategy attempting an outbound socket fails, and the deny is audited."""

from __future__ import annotations

import shutil
import subprocess
import sys

import numpy as np
import pytest

from _synthetic import MOMENTUM, make_panel, package
from enigma.schemas.research import StrategyPackage
from enigma.store import Stores
from enigma.strategy import SandboxJob, SandboxRunner, StrategyRejectedError, symbol_arrays
from enigma.strategy.package import package_content_hash

SOCKET_STRATEGY = """
import numpy as np
import socket
from enigma.strategy.interface import Strategy, StrategyMeta


class Phone(Strategy):
    def required_columns(self):
        return ("perp_close",)

    def metadata(self):
        return StrategyMeta(name="phone_home", version="1")

    def target_positions(self, data):
        socket.create_connection(("1.1.1.1", 443), timeout=2)
        return np.zeros(data["perp_close"].shape[0])
"""


def _socket_package() -> StrategyPackage:
    base = package(MOMENTUM, "Momentum", ("perp_close",))
    return base.model_copy(
        update={
            "source_code": SOCKET_STRATEGY.strip(),
            "entrypoint": "Phone",
            "params": {},
            "content_hash": package_content_hash(
                SOCKET_STRATEGY,
                "Phone",
                {},
                base.interface_version,
                base.cost_model,
                ("perp_close",),
                1,
            ),
        }
    )


@pytest.fixture
def data() -> dict[str, dict[str, np.ndarray]]:
    frame = make_panel(("BTCUSDT",), n_bars=50, seed=1)
    return {"s0": symbol_arrays(frame, "BTCUSDT", ("perp_close",))}


def test_static_precheck_denies_socket_and_audits(
    data: dict[str, dict[str, np.ndarray]], stores: Stores
) -> None:
    pkg = _socket_package()
    with pytest.raises(StrategyRejectedError):
        SandboxRunner(audit=stores.audit).run(pkg, data, [SandboxJob("j", "s0")])
    entries = stores.audit.entries(subject_id=pkg.id)
    assert entries[0].action == "sandbox.denied"
    assert entries[0].details["reason"] == "static_precheck"
    assert any("socket" in v for v in entries[0].details["violations"])


def test_runtime_denies_socket_and_audits(
    data: dict[str, dict[str, np.ndarray]], stores: Stores
) -> None:
    pkg = _socket_package()
    result = SandboxRunner(audit=stores.audit).run(
        pkg, data, [SandboxJob("j", "s0")], static_precheck=False
    )
    assert not result.ok
    assert result.isolation == "netns"
    assert result.violations and result.violations[0]["detail"] == "socket"
    actions = [(e.action, e.details.get("event")) for e in stores.audit.entries(subject_id=pkg.id)]
    assert ("sandbox.denied", "import") in actions
    assert ("sandbox.run", None) in actions
    stores.audit.verify()


def test_network_namespace_blocks_connect_even_without_audit_hook() -> None:
    """The namespace is an independent layer: raw connect() from the same launch is unreachable."""
    unshare = shutil.which("unshare")
    assert unshare is not None
    probe = (
        "import socket, errno\n"
        "s = socket.socket()\n"
        "s.settimeout(3)\n"
        "try:\n"
        "    s.connect(('1.1.1.1', 443))\n"
        "    print('CONNECTED')\n"
        "except OSError as e:\n"
        "    print(errno.errorcode.get(e.errno, e.errno))\n"
    )
    done = subprocess.run(  # noqa: S603
        [unshare, "--user", "--map-root-user", "--net", "--", sys.executable, "-I", "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert done.stdout.strip() == "ENETUNREACH"
