"""Agent-shaped latency benchmark for anon_proxy masking backends.

Simulates agent traffic: N turns, each request carrying the FULL conversation
history (the dominant real-world shape — every turn re-masks the whole
transcript). Measures `adapters.anthropic.mask_request` per turn against the
real privacy-filter model, comparing the torch and onnx backends.

Downloads the model (and, for the onnx arm, the ~0.77 GB q4f16 export), so it
is opt-in:

    ANON_PROXY_LIVE_TESTS=1 uv run --extra onnx python scripts/bench_masking.py
"""

import os
import statistics
import sys
import time

from anon_proxy.adapters import anthropic as ad
from anon_proxy.masker import Masker
from anon_proxy.privacy_filter import PrivacyFilter

N_TURNS = 12

PROSE = (
    "We reviewed the deployment pipeline and the rollout looks stable. "
    "Latency percentiles held under the agreed budget through the canary "
    "window, and the error rate stayed flat across all regions. "
)
CODE = (
    "def rollout(stage, replicas):\n"
    "    for r in range(replicas):\n"
    "        client.patch(f'/deploy/{stage}/{r}', json={'weight': r / replicas})\n"
    "    return client.get(f'/deploy/{stage}/status').json()\n"
)


def user_text(i: int) -> str:
    pii = (
        f" Contact Alice Smith at alice.smith@example.com or 555-867-5309 "
        f"about incident {i}."
        if i % 3 == 0
        else ""
    )
    return f"Turn {i}: " + PROSE * 6 + CODE * 4 + pii


def request_body(n: int) -> dict:
    msgs: list[dict] = []
    for i in range(n + 1):
        msgs.append({"role": "user", "content": user_text(i)})
        if i < n:
            msgs.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"Reply {i}: " + PROSE * 3}],
                }
            )
    return {"model": "claude-x", "messages": msgs}


def run_arm(name: str, pf: PrivacyFilter) -> None:
    m = Masker(filter=pf)
    times: list[float] = []
    for i in range(N_TURNS):
        body = request_body(i)
        t0 = time.perf_counter()
        masked = ad.mask_request(body, m)
        times.append((time.perf_counter() - t0) * 1000)
        assert "alice.smith@example.com" not in str(masked), "PII leaked through mask"
    print(
        f"{name}: cold={times[0]:8.1f}ms  "
        f"warm_median={statistics.median(times[1:]):8.1f}ms  "
        f"warm_p95={sorted(times[1:])[-1]:8.1f}ms  "
        f"total={sum(times):9.1f}ms  store={len(m.store)} entries"
    )


def main() -> None:
    if os.environ.get("ANON_PROXY_LIVE_TESTS") != "1":
        print(
            "This benchmark loads the real model. Re-run with:\n"
            "  ANON_PROXY_LIVE_TESTS=1 uv run --extra onnx "
            "python scripts/bench_masking.py",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"{N_TURNS} turns, full-history per request, real model\n")
    run_arm("torch-cpu ", PrivacyFilter())
    run_arm("onnx-q4f16", PrivacyFilter(backend="onnx"))


if __name__ == "__main__":
    main()
