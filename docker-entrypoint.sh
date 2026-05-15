#!/bin/sh
# Entrypoint: auto-discover config files in /config and exec the proxy.
#
# Auto-pick patterns.json and merge_gap.json from /config when present, unless
# the user has set the corresponding env var explicitly. Capture and metrics
# are opt-in via env vars (ANON_PROXY_CAPTURE, ANON_PROXY_METRICS) and are not
# auto-configured here so we don't unintentionally write UNMASKED PII to disk.

set -e

if [ -z "${ANON_PROXY_PATTERNS}" ] && [ -f "/config/patterns.json" ]; then
    export ANON_PROXY_PATTERNS=/config/patterns.json
fi

if [ -z "${ANON_PROXY_MERGE_GAP}" ] && [ -f "/config/merge_gap.json" ]; then
    export ANON_PROXY_MERGE_GAP=/config/merge_gap.json
fi

exec python -m anon_proxy.server "$@"
