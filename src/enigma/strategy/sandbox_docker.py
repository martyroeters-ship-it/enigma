"""Docker Linux backend for the strategy sandbox.

Used only when this host cannot create a Linux user and network namespace
(`unshare`). The container has no network, no Docker socket, and no mount
except the one temporary directory that holds the strategy source and the
arrays it is allowed to read. Ollama and the institution database stay on
the host.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

SANDBOX_IMAGE = "enigma-sandbox:py312"
_PROBE = (
    "import os, socket, sys\n"
    "import numpy\n"
    "if os.path.exists('/var/run/docker.sock'):\n"
    "    sys.exit(3)\n"
    "sock = socket.socket()\n"
    "sock.settimeout(2)\n"
    "try:\n"
    "    sock.connect(('1.1.1.1', 443))\n"
    "except OSError:\n"
    "    sys.exit(0)\n"
    "sys.exit(2)\n"
)


class _ProbeCache:
    status: tuple[bool, str] | None = None


def reset_docker_backend_status() -> None:
    _ProbeCache.status = None


def docker_run_argv(
    box: Path,
    *,
    name: str,
    cpu_seconds: int,
    memory_bytes: int,
    open_files: int,
    file_size_bytes: int,
) -> list[str]:
    """Argv for one strategy container. No socket mount and no host environment."""
    source = str(box.resolve())
    if any(ch in source for ch in ",\n\r"):
        raise ValueError("refusing to mount a sandbox path that contains a mount separator")
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        "64",
        "--memory",
        str(memory_bytes),
        "--memory-swap",
        str(memory_bytes),
        "--cpus",
        "1",
        "--ulimit",
        f"cpu={cpu_seconds}:{cpu_seconds}",
        "--ulimit",
        f"nofile={open_files}:{open_files}",
        "--ulimit",
        f"fsize={file_size_bytes}:{file_size_bytes}",
        "--ulimit",
        "core=0:0",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--mount",
        f"type=bind,source={source},target=/work",
        "--workdir",
        "/work",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",  # noqa: S108 - container mount, not a host path
        "-e",
        "HOME=/work",
        "-e",
        "TMPDIR=/work",
        "-e",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "-e",
        "LANG=C.UTF-8",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "OPENBLAS_NUM_THREADS=1",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "MKL_NUM_THREADS=1",
        SANDBOX_IMAGE,
        "python",
        "-I",
        "-B",
        "/work/_bootstrap.py",
    ]


def _docker_bin() -> str | None:
    return shutil.which("docker")


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def docker_backend_status(*, refresh: bool = False) -> tuple[bool, str]:
    """Whether a network-disabled Linux container can execute strategies.

    A successful probe, or a probe that proves the image is not isolated, is cached
    for this process. A missing daemon or image is not cached, so starting Docker
    and building the image is picked up by the next check.
    """
    if _ProbeCache.status is not None and not refresh:
        return _ProbeCache.status
    result = _probe_docker()
    if result[0] or result[1].startswith("Image "):
        _ProbeCache.status = result
    return result


def _probe_docker() -> tuple[bool, str]:
    docker = _docker_bin()
    if docker is None:
        return False, "The docker command is not installed."
    try:
        info = _run([docker, "info"], timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"The Docker daemon did not answer ({exc})."
    if info.returncode != 0:
        detail = (info.stderr or info.stdout or "docker info failed").strip()
        return False, f"The Docker daemon is not running ({detail[:240]})."
    image = _run([docker, "image", "inspect", "--format", "{{.Id}}", SANDBOX_IMAGE], timeout=20)
    if image.returncode != 0:
        return (
            False,
            f"The local image {SANDBOX_IMAGE} is not built. Run `uv run enigma sandbox prepare`.",
        )
    try:
        probe = _run(
            [
                docker,
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--pids-limit",
                "32",
                "--memory",
                "512m",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-e",
                "PYTHONDONTWRITEBYTECODE=1",
                SANDBOX_IMAGE,
                "python",
                "-I",
                "-B",
                "-c",
                _PROBE,
            ],
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"The sandbox image probe failed ({exc})."
    if probe.returncode == 2:
        return False, f"Image {SANDBOX_IMAGE} was able to open a network connection."
    if probe.returncode == 3:
        return False, f"Image {SANDBOX_IMAGE} can see the Docker socket."
    if probe.returncode != 0:
        tail = (probe.stderr or probe.stdout or "").strip()
        return False, f"Image {SANDBOX_IMAGE} failed its probe ({tail[:240]})."
    return True, f"Linux container {SANDBOX_IMAGE}, network disabled"


def isolation_unavailable_message(detail: str) -> str:
    """Tell the operator how to get a real sandbox. Do not suggest weak mode."""
    if platform.system() == "Darwin":
        where = (
            "macOS cannot create the Linux network namespace this sandbox uses. "
            "Strategy code runs in a Linux container; Ollama, the dashboard, and the "
            "orchestrator stay on macOS."
        )
    else:
        where = "Linux network namespaces (unshare --user --map-root-user --net) are not usable."
    return (
        "strategy isolation is not available. "
        f"{where} {detail} "
        "Install Docker Desktop, start it, and run `uv run enigma sandbox prepare`. "
        "Refusing to execute strategy code."
    )


def prepare_sandbox_image() -> str:
    """Build the local sandbox image and refuse it unless the network probe fails closed."""
    docker = _docker_bin()
    if docker is None:
        raise RuntimeError("The docker command is not installed.")
    dockerfile = Path(__file__).resolve().parents[3] / "deploy" / "sandbox" / "Dockerfile"
    if not dockerfile.is_file():
        raise RuntimeError(f"Sandbox Dockerfile is missing at {dockerfile}.")
    build = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [docker, "build", "-t", SANDBOX_IMAGE, "-f", str(dockerfile), str(dockerfile.parent)],
        check=False,
    )
    if build.returncode != 0:
        raise RuntimeError(f"docker build of {SANDBOX_IMAGE} failed.")
    reset_docker_backend_status()
    ok, detail = docker_backend_status(refresh=True)
    if not ok:
        raise RuntimeError(detail)
    return detail


def spawn_docker(
    box: Path,
    parent_dir: Path,
    *,
    name_seed: str,
    wall_seconds: float,
    cpu_seconds: int,
    memory_bytes: int,
    open_files: int,
    file_size_bytes: int,
) -> tuple[int, bool]:
    """Run the prepared box. Returns (exit code, timed out)."""
    docker = _docker_bin() or "docker"
    name = "enigma-sbx-" + "".join(ch if ch.isalnum() else "" for ch in name_seed)[:40]
    argv = docker_run_argv(
        box,
        name=name,
        cpu_seconds=cpu_seconds,
        memory_bytes=memory_bytes,
        open_files=open_files,
        file_size_bytes=file_size_bytes,
    )
    timed_out = False
    with (
        (parent_dir / "stdout.log").open("wb") as out,
        (parent_dir / "stderr.log").open("wb") as err,
    ):
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            argv,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
            close_fds=True,
        )
        deadline = time.monotonic() + wall_seconds
        while proc.poll() is None:
            if time.monotonic() > deadline:
                timed_out = True
                subprocess.run(  # noqa: S603
                    [docker, "kill", name],
                    capture_output=True,
                    check=False,
                )
                proc.wait(timeout=30)
                break
            time.sleep(0.05)
    subprocess.run(  # noqa: S603 - remove the container if --rm lost the race
        [docker, "rm", "-f", name],
        capture_output=True,
        check=False,
    )
    code = proc.returncode if proc.returncode is not None else -9
    return int(code), timed_out
