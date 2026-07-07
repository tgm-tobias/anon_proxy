"""Poll the proxy's /_status endpoint. Any failure means "down"."""

from __future__ import annotations

import httpx


def fetch_status(url: str, *, get=None, timeout: float = 2.0) -> dict | None:
    get = get or httpx.get
    try:
        resp = get(url, timeout=timeout)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
