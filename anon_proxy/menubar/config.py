"""Persisted menu-bar preferences."""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULTS: dict = {
    "theme": "auto",
    "start_at_login": False,
    "url": "http://127.0.0.1:8080/_status",
}


def default_path() -> Path:
    return Path.home() / ".config" / "anon-proxy" / "menubar.json"


def load_config(path: Path | None = None) -> dict:
    path = path or default_path()
    cfg = dict(DEFAULTS)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return cfg
    if isinstance(data, dict):
        cfg.update({key: data[key] for key in DEFAULTS if key in data})
    return cfg


def save_config(cfg: dict, path: Path | None = None) -> None:
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)
