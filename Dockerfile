# anon-proxy: PII masking proxy for LLM APIs (CPU-only image for k8s)
#
# Build (BuildKit required — default on modern Docker):
#   docker build -t anon-proxy:latest .
#
# Run (k8s pod sketch):
#   volumes:
#     - name: config
#       configMap: { name: anon-proxy-config }   # config.json
#     - name: data
#       persistentVolumeClaim: { claimName: anon-proxy-data }
#     - name: models
#       persistentVolumeClaim: { claimName: anon-proxy-models }   # HF cache (~500MB)
#   volumeMounts:
#     - { name: config, mountPath: /config, readOnly: true }
#     - { name: data,   mountPath: /data }
#     - { name: models, mountPath: /models }
#   env:
#     - { name: ANON_PROXY_METRICS, value: "true" }
#     - { name: ANON_PROXY_CAPTURE, value: "/data/capture.jsonl" }   # optional; UNMASKED PII
#     - { name: ANON_PROXY_STORE,   value: "/data/pii_store.json" }  # default in the image; overridable
#
# /config and /data are wired up by /docker-entrypoint.sh — you only need to
# set the env vars you want.
#
# Model weights are NOT baked into the image. HF_HOME points at /models, so
# on first start the pod downloads openai/privacy-filter into the PVC; every
# subsequent start (and every other replica that mounts the same PVC) reuses
# the cached files. Expect a one-time stall on the first request after a
# fresh PVC. To pre-populate, run once locally with HF_HOME pointing at the
# volume and rsync the result up. The default onnx backend fetches the q4f16
# graph plus its weights sidecar (~0.77 GB); size the models PVC accordingly.

# ---------------------------------------------------------------------------
# Builder: resolve and install into /app/.venv from uv.lock.
# ---------------------------------------------------------------------------
# torch resolves to the CPU-only wheel here — pyproject pins the pytorch-cpu
# index for sys_platform == 'linux', which keeps the ~3 GB of CUDA libraries
# out of the image. See the [tool.uv.sources] comment in pyproject.toml.
#
# --extra onnx adds onnxruntime, for ANON_PROXY_BACKEND=onnx (the image
# default): the pre-quantized q4f16 graph is ~9x faster than torch on CPU.
FROM ghcr.io/astral-sh/uv:0.8-python3.10-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /src

# 1. Dependencies only, from the lock. Split from the project install so this
#    layer (the slow one — torch and friends) survives source-only edits.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --extra onnx --no-install-project

# 2. The project itself. --no-editable copies it into site-packages so the
#    runtime stage needs nothing but the venv.
COPY pyproject.toml uv.lock README.md ./
COPY anon_proxy ./anon_proxy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra onnx --no-editable

# ---------------------------------------------------------------------------
# Runtime: the venv, and nothing else. No uv, no build tooling, no sources.
# ---------------------------------------------------------------------------
FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    ANON_PROXY_HOST=0.0.0.0 \
    ANON_PROXY_PORT=8080 \
    ANON_PROXY_BACKEND=onnx \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv

WORKDIR /app

# Mount points for runtime configuration, persistent capture/metrics output,
# and the HF model cache.
#   /config: read-only ConfigMap with config.json
#   /data:   read-write PVC for capture.jsonl
#   /models: read-write PVC for HF_HOME — populated on first run, reused thereafter
VOLUME ["/config", "/data", "/models"]

EXPOSE 8080

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
