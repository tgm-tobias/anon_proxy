#!/bin/sh
# Entrypoint: auto-discover the config file in /config and exec the proxy.
#
# Auto-pick config.json from /config when present, unless the user has set
# ANON_PROXY_CONFIG explicitly. Capture and metrics are opt-in via env vars
# (ANON_PROXY_CAPTURE, ANON_PROXY_METRICS) and are not auto-configured here
# so we don't unintentionally write UNMASKED PII to disk.

set -e

if [ -z "${ANON_PROXY_CONFIG}" ] && [ -f "/config/config.json" ]; then
    export ANON_PROXY_CONFIG=/config/config.json
fi

exec python -m anon_proxy.server "$@"
