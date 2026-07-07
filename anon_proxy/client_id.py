"""Classify the calling agent from request headers.

The classifier stores only the resulting label. It never stores raw headers,
which may include credentials.
"""

from __future__ import annotations


def classify_client(headers: dict[str, str]) -> str:
    ua = headers.get("user-agent", "")
    ua_l = ua.lower()
    originator = headers.get("originator", "").lower()

    if "claude-cli" in ua_l:
        return "Claude Code"
    if "codex" in originator or "codex" in ua_l:
        return "Codex"

    has_stainless = any(k.startswith("x-stainless") for k in headers)
    if has_stainless or "anthropic-version" in headers:
        pkg = headers.get("x-stainless-package-version", "").lower()
        if "anthropic-version" in headers or "anthropic" in ua_l or "anthropic" in pkg:
            return "Anthropic SDK"
        return "OpenAI SDK"

    if ua:
        token = ua.split("/", 1)[0].strip()
        return token or "unknown"
    return "unknown"
