# Local startup

This procedure was checked on a Linux VM (Python 3.12, the repo virtualenv, Ollama pid left running). It was not run on macOS. The commands are the same ones the Mac operator runs. Where a step is Mac-specific, it says so.

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

## Start

```bash
export ENIGMA_HOME="$HOME/enigma-run"
export ENIGMA_MAX_DAILY_API_SPEND_USD=0
unset ENIGMA_PAPER_ACK_ID ENIGMA_HYPOTHESIS_MODE
uv run enigma init
uv run enigma run --until-idle
```

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
