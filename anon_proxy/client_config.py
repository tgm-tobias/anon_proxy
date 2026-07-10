"""Client base-URL configuration helpers for anon-proxy."""

from __future__ import annotations

from pathlib import Path

PROVIDERS: dict[str, dict[str, str]] = {
    "claude": {"env_var": "ANTHROPIC_BASE_URL", "path": "/anthropic"},
    "codex": {"env_var": "OPENAI_BASE_URL", "path": "/openai"},
    "openai": {"env_var": "OPENAI_BASE_URL", "path": "/openai"},
}


def _provider_config(provider: str) -> dict[str, str]:
    try:
        return PROVIDERS[provider]
    except KeyError as e:
        raise ValueError(f"unknown provider {provider!r}") from e


def base_url_for(provider: str, host: str = "127.0.0.1", port: int = 8080) -> str:
    cfg = _provider_config(provider)
    return f"http://{host}:{port}{cfg['path']}"


def env_snippet(provider: str, url: str) -> str:
    cfg = _provider_config(provider)
    return f"export {cfg['env_var']}={url}"


def apply_env(provider: str, url: str, rc_path: Path) -> None:
    line = env_snippet(provider, url)
    existing = rc_path.read_text() if rc_path.exists() else ""
    if line in existing:
        return
    separator = "" if not existing or existing.endswith("\n") else "\n"
    rc_path.write_text(f"{existing}{separator}{line}\n")
