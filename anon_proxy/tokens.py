"""Output-token measurement helpers."""

from __future__ import annotations


def approx_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def extract_output_tokens(adapter_name: str, resp: dict) -> int | None:
    usage = resp.get("usage") if isinstance(resp, dict) else None
    if not isinstance(usage, dict):
        return None
    key = "output_tokens" if adapter_name == "anthropic" else "completion_tokens"
    value = usage.get(key)
    return int(value) if isinstance(value, (int, float)) else None
