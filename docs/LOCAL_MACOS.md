# Local startup

This procedure was checked on a Linux VM for the orchestrator, Ollama, and the Linux `unshare` sandbox. The Docker backend below is the macOS path. It was not executed on the MacBook Air from this environment. Where a step is Mac-specific, it says so.

Paper trading and live trading stay off. Do not copy `registries/risk/limits-paper-enabled.yaml` over `registries/risk/limits.yaml`. Do not set `ENIGMA_PAPER_ACK_ID`. Do not set `ENIGMA_HOST_ROLE` unless you are splitting hosts; the default `single` process is the one-machine institution and is not an M5 acceptance.

## What you need

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- One Ollama server. Do not start a second `ollama serve` if one is already running.
- Model `qwen2.5:7b-instruct`. Optional: `qwen2.5:3b` for a benchmark comparison. Neither is adopted by a benchmark score alone.
- Disk: the 7B blob is about 5 GB and the optional 3B is smaller. Leave several extra GB for market archives under `ENIGMA_HOME`.
- RAM: the 7B runner used about 7 GB RSS on this VM at the server's default context. A longer `num_ctx` uses more. Close other heavy apps before a hypothesis call.
- A hypothesis call on the 7B model has taken from about 6 minutes to about 20 minutes. The dashboard heartbeat now updates during that call.

## Install

```bash
git clone https://github.com/martyroeters-ship-it/enigma.git
cd enigma
uv sync
```

On a Mac, install Ollama from https://ollama.com and pull the model in a terminal:

```bash
ollama pull qwen2.5:7b-instruct
# optional
ollama pull qwen2.5:3b
```

If nothing is listening on port 11434:

```bash
OLLAMA_NUM_PARALLEL=1 ollama serve
```

`OLLAMA_NUM_PARALLEL=1` keeps one generation on the loaded weights. If a server is already running, leave it.

## Strategy sandbox on a Mac

Strategy code does not run in the macOS Python process. Linux hosts use a network namespace. A Mac uses Docker Desktop to run the same bootstrap in a Linux container with network disabled. Ollama, the dashboard, and the research process stay native.

Do not set `ENIGMA_SANDBOX_ALLOW_WEAK`. That switch is not accepted by `enigma up`, `enigma research`, `enigma run`, or `enigma resume`.

1. Install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/) and open it. Wait until the engine is running. Apple Silicon and Intel both use the multi-arch `python:3.12-slim-bookworm` image; `prepare` builds for the machine you are on.
2. Confirm the daemon from a terminal:

```bash
docker info
```

3. Update the code checkout. Leave `~/enigma-run` in place; that directory is the institutional memory and is not rebuilt by these commands.

```bash
cd ~/enigma
git fetch origin
git checkout main
git pull
uv sync
unset ENIGMA_SANDBOX_ALLOW_WEAK
uv run enigma sandbox prepare
uv run enigma sandbox status
```

`prepare` needs the network once, to pull the base image and NumPy into a local image tagged `enigma-sandbox:py312`. Later research cycles do not pull or build. `status` prints `docker` when the container probe proves that outbound `connect()` fails and `/var/run/docker.sock` is not in the container. If it prints an error, start Docker Desktop and run `prepare` again. Do not continue to a research cycle.

The container is read-only except for one temporary directory, drops every capability, sets no-new-privileges, limits CPU, memory, processes, and file size, and does not receive host environment variables. It is not given the Docker socket, `ENIGMA_HOME`, credentials, or the sealed validation store. Only the arrays for that trial are copied in.

## Start

```bash
export ENIGMA_HOME="$HOME/enigma-run"
export ENIGMA_MAX_DAILY_API_SPEND_USD=0
unset ENIGMA_PAPER_ACK_ID
unset ENIGMA_SANDBOX_ALLOW_WEAK
uv run enigma init
uv run enigma sandbox status
uv run enigma up
```

On a Mac, `uv run enigma sandbox status` must print `docker` before `up`. On Linux it prints `netns`.

`enigma up` refuses to start unless the daily spend cap is $0, both trading flags in `registries/risk/limits.yaml` are false, and strategy isolation is `netns` or `docker`. It checks that Ollama answers on `http://127.0.0.1:11434`, sets `ENIGMA_HYPOTHESIS_MODE=llm` when that variable is unset, takes the home's daemon lock, requeues orphaned tasks, and prints a line after each step. A second `enigma up` or `enigma run` against the same home exits instead of writing beside the first process. `Ctrl-C` finishes the current step.

One research cycle, after the sandbox status line says `docker`:

```bash
export ENIGMA_HOME="$HOME/enigma-run"
export ENIGMA_MAX_DAILY_API_SPEND_USD=0
unset ENIGMA_PAPER_ACK_ID
unset ENIGMA_SANDBOX_ALLOW_WEAK
uv run enigma research
```

A validation FAIL is a finished cycle. An isolation error is not. If isolation is down, the cycle stops before code generation and does not spend three model repairs on it.

Hypothesis calls on the 7B model can run for many minutes. The Ollama provider timeout in `registries/models/models.yaml` is 1800 seconds.

In a second terminal, the read-only dashboard:

```bash
export ENIGMA_HOME="$HOME/enigma-run"
uv run enigma dashboard --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/ . JSON lives at `/api/health`, `/api/cycles`, `/api/learning`, `/api/costs`, `/api/models`, `/api/agents`, `/api/knowledge`.

`ENIGMA_VALIDATION_PROTOCOL=1.0.0` opts that process into the full gate set. It does not change `PROTOCOL_VERSION` in the source, which stays `0.1.0-slice`.

## Stop

`enigma run` handles SIGINT and SIGTERM. The current step finishes, then the heartbeat is `stopped`.

```bash
kill -TERM <daemon-pid>
```

Stop the dashboard with SIGTERM on its pid. Do not kill the Ollama server unless you intend to shut the model down.

## Monitor

- `uv run enigma status` reads `ENIGMA_HOME` and prints the queue, state machines, costs, and audit-chain check.
- `uv run enigma cycle <id>` prints one cycle as JSON.
- A worker whose process has exited is `stopped` or `stopped (process exited)`. `stale (no heartbeat)` means the pid is still alive and has not written a heartbeat for a minute.
- `uv run enigma resume` requeues tasks whose worker died.
- `uv run enigma kill-switch on` halts agent work. `off` allows it again.

## Credentials

`uv run enigma credentials` lists optional variable names and whether they are set. It does not print values. No key is required for Ollama or for public market archives. Leave `ENIGMA_MAX_DAILY_API_SPEND_USD` at 0 unless you intentionally fund a named provider. `TYPESAFE_API_KEY` is optional; if it is unset, TypeSafe is not measured and nothing is adopted.
