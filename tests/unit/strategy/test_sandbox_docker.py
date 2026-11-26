"""The Docker sandbox argv is a closed Linux container. No daemon is required."""

from __future__ import annotations

from pathlib import Path

import pytest

from enigma.strategy.sandbox_docker import docker_run_argv, isolation_unavailable_message


def test_docker_argv_closes_network_credentials_and_the_socket(tmp_path: Path) -> None:
    box = tmp_path / "box"
    box.mkdir()
    argv = docker_run_argv(
        box,
        name="enigma-sbx-test",
        cpu_seconds=120,
        memory_bytes=4 * 1024**3,
        open_files=64,
        file_size_bytes=512 * 1024**2,
    )
    text = " ".join(argv)
    assert argv[0] == "docker"
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert "--read-only" in argv
    assert "--cap-drop" in argv and argv[argv.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges:true" in argv
    assert "--pids-limit" in argv and argv[argv.index("--pids-limit") + 1] == "64"
    assert "--memory" in argv and "--memory-swap" in argv
    assert "--cpus" in argv and argv[argv.index("--cpus") + 1] == "1"
    assert any(item.startswith("cpu=120:120") for item in argv)
    assert "--user" in argv
    mount = argv[argv.index("--mount") + 1]
    assert f"source={box.resolve()}" in mount
    assert "target=/work" in mount
    assert argv.count("--mount") == 1
    assert "docker.sock" not in text
    assert "--privileged" not in argv
    assert "ENIGMA_HOME" not in text
    assert argv[-4:] == ["python", "-I", "-B", "/work/_bootstrap.py"]
    assert "-S" not in argv
    assert "enigma-sandbox:py312" in argv


def test_docker_argv_rejects_a_mount_separator_in_the_path(tmp_path: Path) -> None:
    box = tmp_path / "a,b"
    box.mkdir()
    with pytest.raises(ValueError, match="mount separator"):
        docker_run_argv(
            box,
            name="enigma-sbx-test",
            cpu_seconds=1,
            memory_bytes=1024,
            open_files=64,
            file_size_bytes=1024,
        )


def test_macos_message_names_docker_and_does_not_offer_weak_mode(
    monkeypatch,
) -> None:
    monkeypatch.setattr("enigma.strategy.sandbox_docker.platform.system", lambda: "Darwin")
    text = isolation_unavailable_message("The docker command is not installed.")
    assert "macOS cannot create the Linux network namespace" in text
    assert "Docker Desktop" in text
    assert "uv run enigma sandbox prepare" in text
    assert "ALLOW_WEAK" not in text
    assert "Ollama" in text
