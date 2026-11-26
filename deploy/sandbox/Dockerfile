# Linux image for strategy execution. The orchestrator, Ollama, and the dashboard
# stay on the host. This image is used only when Linux unshare is unavailable
# (macOS). Build it with `uv run enigma sandbox prepare`. Strategy runs do not
# pull or build anything.
FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir "numpy==2.2.6" \
    && pip uninstall -y pip \
    && rm -rf /root/.cache /usr/local/bin/pip /usr/local/bin/pip3

RUN useradd --uid 65532 --create-home --shell /usr/sbin/nologin sandbox
USER sandbox
WORKDIR /work
