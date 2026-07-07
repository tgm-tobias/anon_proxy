"""Dino skins: theme registry, holiday calendar, and asset resolution."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

FRAMES: tuple[str, ...] = ("stand", "run1", "run2", "dead", "cactus")

THEMES: dict[str, str] = {
    "classic": "classic",
    "winter": "winter",
    "halloween": "halloween",
}

_ASSETS = Path(__file__).resolve().parent.parent / "assets" / "dino"


def holiday_for(date: dt.date) -> str:
    month, day = date.month, date.day
    if (month == 12 and day >= 20) or (month == 1 and day == 1):
        return "winter"
    if month == 10 and day >= 24:
        return "halloween"
    return "classic"


def resolve_theme(selected: str, date: dt.date) -> str:
    name = holiday_for(date) if selected == "auto" else selected
    return name if name in THEMES else "classic"


def frame_paths(theme: str, *, base: Path | None = None) -> dict[str, Path]:
    base = base if base is not None else _ASSETS
    subdir = THEMES.get(theme, theme)
    theme_dir = base / subdir
    if not theme_dir.exists():
        theme_dir = base / "classic"
    classic_dir = base / "classic"
    paths: dict[str, Path] = {}
    for frame in FRAMES:
        candidate = theme_dir / f"{frame}.png"
        paths[frame] = candidate if candidate.exists() else classic_dir / f"{frame}.png"
    return paths
