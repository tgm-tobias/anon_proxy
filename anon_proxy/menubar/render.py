"""Pure status-to-presentation logic for the menu-bar dino."""

from __future__ import annotations

from dataclasses import dataclass


def fps_for(tps: float) -> float:
    return min(12.0, 1.5 + tps / 28.0)


@dataclass
class RenderState:
    icon_state: str
    fps: float
    title: str
    tooltip: str
    menu: list[str]


def _fmt_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def _menu_lines(status: dict) -> list[str]:
    tps = int(round(float(status.get("tokens_per_sec", 0.0) or 0.0)))
    lines = [
        f"Running - {status.get('listen_addr') or '?'} - {tps} tok/s",
        f"Driving: {status.get('last_client') or '-'}",
        f"Requests {_fmt_int(status.get('requests_masked_total'))}"
        f" - PII {_fmt_int(status.get('entities_masked_total'))}"
        f" - tokens {_fmt_int(status.get('tokens_out_total'))}",
    ]
    by_client = status.get("by_client") or {}
    if by_client:
        parts = [
            f"{name} {_fmt_int(values.get('requests'))}"
            for name, values in by_client.items()
            if isinstance(values, dict)
        ]
        if parts:
            lines.append("By agent: " + " - ".join(parts))
    errors = int(status.get("masking_errors_total", 0) or 0)
    if errors:
        lines.append(f"Masking errors: {errors}")
    lines.append(
        f"Backend: {status.get('backend') or '?'} - Store: {_fmt_int(status.get('store'))}"
    )
    return lines


def render(status: dict | None, *, alarm: bool, now: float) -> RenderState:
    if status is None:
        return RenderState(
            icon_state="down",
            fps=0.0,
            title="",
            tooltip="anon-proxy: not running",
            menu=["Not running"],
        )

    tps = float(status.get("tokens_per_sec", 0.0) or 0.0)
    if alarm:
        icon_state = "alarm"
    elif tps > 0.0:
        icon_state = "running"
    else:
        icon_state = "idle"

    title = str(int(round(tps))) if icon_state == "running" else ""
    driving = status.get("last_client") or "-"
    tooltip = (
        "anon-proxy: MASKING ERROR - check the proxy"
        if icon_state == "alarm"
        else f"anon-proxy: {int(round(tps))} tok/s - {driving}"
    )
    return RenderState(
        icon_state=icon_state,
        fps=fps_for(tps),
        title=title,
        tooltip=tooltip,
        menu=_menu_lines(status),
    )


def format_watch_line(status: dict | None, *, alarm: bool, now: float) -> str:
    state = render(status, alarm=alarm, now=now)
    if state.icon_state == "down":
        return "down - proxy not running"
    glyph = {"running": "run", "idle": "idle", "alarm": "alarm"}[state.icon_state]
    suffix = "" if state.icon_state == "running" else f" [{state.icon_state}]"
    return f"{glyph} {state.menu[0]}{suffix}"
