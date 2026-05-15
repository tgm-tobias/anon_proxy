# anon-proxy: PII masking proxy for LLM APIs (CPU-only image for k8s)
#
# Build:
#   docker build -t anon-proxy:latest .
#
# Run (k8s pod sketch):
#   volumes:
#     - name: config
#       configMap: { name: anon-proxy-config }   # patterns.json, merge_gap.json
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
#
# /config and /data are wired up by /docker-entrypoint.sh — you only need to
# set the env vars you want.
#
# Model weights are NOT baked into the image. HF_HOME points at /models, so
# on first start the pod downloads openai/privacy-filter into the PVC; every
# subsequent start (and every other replica that mounts the same PVC) reuses
# the cached files. Expect a one-time stall on the first request after a
# fresh PVC. To pre-populate, run once locally with HF_HOME pointing at the
# volume and rsync the result up.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    ANON_PROXY_HOST=0.0.0.0 \
    ANON_PROXY_PORT=8080 \
    ANON_PROXY_BACKEND=cpu

WORKDIR /app

# 1. CPU-only torch wheel from PyTorch's index. Saves ~3 GB vs PyPI's
#    CUDA-bundled Linux wheel. Pinning by major version satisfies the
#    `torch>=2.11.0` constraint in pyproject.toml.
#    --index-url makes the CPU index primary (so torch's `+cpu` build wins over
#    PyPI's CUDA wheel); --extra-index-url adds PyPI as fallback for transitive
#    deps like typing-extensions that aren't mirrored on the PyTorch index.
RUN pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch>=2.11.0"

# 2. Project metadata + source. Copy in two steps so the Docker layer cache
#    survives source-only edits when pyproject is unchanged.
COPY pyproject.toml README.md ./
COPY anon_proxy ./anon_proxy

# 3. Install the project itself + remaining deps. torch is already installed
#    from the CPU index above, so pip skips it here.
RUN pip install .

# 4. Mount points for runtime configuration, persistent capture/metrics output,
#    and the HF model cache.
#    /config: read-only ConfigMap with patterns.json / merge_gap.json
#    /data:   read-write PVC for capture.jsonl
#    /models: read-write PVC for HF_HOME — populated on first run, reused thereafter
VOLUME ["/config", "/data", "/models"]

EXPOSE 8080

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
