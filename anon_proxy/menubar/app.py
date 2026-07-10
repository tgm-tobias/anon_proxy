"""Menu-bar entry point for the anon-proxy dino indicator."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

from anon_proxy import client_config
from anon_proxy.menubar import config as cfg
from anon_proxy.menubar import themes
from anon_proxy.menubar.render import format_watch_line, render
from anon_proxy.menubar.statusclient import fetch_status
from anon_proxy.menubar.supervisor import (
    ProxySupervisor,
    install_launch_agent,
    uninstall_launch_agent,
)

_LABEL = "com.anon-proxy.menubar"


class AlarmLatch:
    """Latch when masking_errors_total rises above the last reset baseline."""

    def __init__(self) -> None:
        self._baseline: int | None = None
        self._latched = False

    def update(self, status: dict | None) -> bool:
        if not status:
            return self._latched
        errors = int(status.get("masking_errors_total", 0) or 0)
        if self._baseline is None:
            self._baseline = errors
        if errors > self._baseline:
            self._latched = True
        return self._latched

    def reset(self) -> None:
        self._baseline = None
        self._latched = False


def watch_once(url: str, *, alarm: bool, now: float, get=None) -> str:
    status = fetch_status(url, get=get)
    return format_watch_line(status, alarm=alarm, now=now)


def watch_loop(url: str, *, interval: float = 2.0) -> None:
    latch = AlarmLatch()
    print(f"watching {url} (Ctrl-C to stop)")
    try:
        while True:
            status = fetch_status(url)
            alarm = latch.update(status)
            print(format_watch_line(status, alarm=alarm, now=time.time()))
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped.")


def _run_macos_app(
    url: str, *, start_proxy: bool = False, backend: str | None = None
) -> None:
    import rumps

    class DinoApp(rumps.App):
        def __init__(self) -> None:
            super().__init__("anon-proxy", quit_button="Quit")
            self._cfg = cfg.load_config()
            self._url = url or self._cfg["url"]
            self._latch = AlarmLatch()
            self._supervisor = ProxySupervisor(backend=backend)
            self._last_status: dict | None = None
            self._frame_idx = 0
            self._last_frame_at = 0.0
            self._last_poll_at = 0.0
            self._icon_frames = self._load_frames()
            self._theme_items: dict[str, object] = {}
            self._status_items: list[object] = []
            self._start_at_login_item = None
            self._build_menu()
            if start_proxy:
                self._supervisor.start()
            rumps.Timer(self._tick, 0.1).start()

        def _load_frames(self) -> dict:
            theme = themes.resolve_theme(self._cfg["theme"], dt.date.today())
            return themes.frame_paths(theme)

        def _save_cfg(self) -> None:
            cfg.save_config(self._cfg)

        def _build_menu(self) -> None:
            self.menu.clear()
            self._theme_items = {}
            self._status_items = [
                rumps.MenuItem(f"status {idx}", callback=None) for idx in range(5)
            ]
            for item in self._status_items:
                self.menu.add(item)
            self.menu.add(None)
            theme_menu = rumps.MenuItem("Theme")
            for name in ("auto", "classic", "halloween", "winter"):
                item = rumps.MenuItem(name.title(), callback=self._set_theme)
                item.state = 1 if self._cfg["theme"] == name else 0
                theme_menu.add(item)
                self._theme_items[name] = item
            self.menu.add(theme_menu)
            self.menu.add(rumps.MenuItem("Reset alarm", callback=self._reset_alarm))
            self.menu.add(None)
            self.menu.add(rumps.MenuItem("Start proxy", callback=self._start_proxy))
            self.menu.add(rumps.MenuItem("Stop proxy", callback=self._stop_proxy))
            self.menu.add(rumps.MenuItem("Restart proxy", callback=self._restart_proxy))
            self.menu.add(None)
            self.menu.add(
                rumps.MenuItem(
                    "Copy Claude Code base URL", callback=self._copy_claude_url
                )
            )
            self.menu.add(
                rumps.MenuItem("Copy OpenAI base URL", callback=self._copy_openai_url)
            )
            self.menu.add(None)
            item = rumps.MenuItem(
                "Start at login", callback=self._toggle_start_at_login
            )
            item.state = 1 if self._cfg["start_at_login"] else 0
            self._start_at_login_item = item
            self.menu.add(item)
            self.menu.add(None)

        def _set_theme(self, sender) -> None:
            selected = str(sender.title).lower()
            self._cfg["theme"] = selected
            self._save_cfg()
            self._icon_frames = self._load_frames()
            for name, item in self._theme_items.items():
                item.state = 1 if name == selected else 0

        def _toggle_start_at_login(self, _sender) -> None:
            enabled = not bool(self._cfg["start_at_login"])
            args = [sys.executable, "-m", "anon_proxy.menubar.app", "--url", self._url]
            if enabled:
                install_launch_agent(_LABEL, args)
            else:
                uninstall_launch_agent(_LABEL)
            self._cfg["start_at_login"] = enabled
            self._save_cfg()
            if self._start_at_login_item is not None:
                self._start_at_login_item.state = 1 if enabled else 0

        def _reset_alarm(self, _sender) -> None:
            self._latch.reset()

        def _start_proxy(self, _sender) -> None:
            self._supervisor.start()

        def _stop_proxy(self, _sender) -> None:
            self._supervisor.stop()

        def _restart_proxy(self, _sender) -> None:
            self._supervisor.restart()

        def _copy_url(self, provider: str) -> None:
            url = client_config.base_url_for(provider)
            try:
                from AppKit import NSPasteboard, NSStringPboardType

                pb = NSPasteboard.generalPasteboard()
                pb.declareTypes_owner_([NSStringPboardType], None)
                pb.setString_forType_(url, NSStringPboardType)
                rumps.notification("anon-proxy", "Copied", url)
            except Exception:
                print(url)

        def _copy_claude_url(self, _sender) -> None:
            self._copy_url("claude")

        def _copy_openai_url(self, _sender) -> None:
            self._copy_url("openai")

        def _tick(self, _timer) -> None:
            now = time.time()
            if now - self._last_poll_at >= 2.0:
                self._last_status = fetch_status(self._url)
                self._last_poll_at = now
            alarm = self._latch.update(self._last_status)
            state = render(self._last_status, alarm=alarm, now=now)
            self._animate(state, now)
            self.title = f" {state.title}" if state.title else ""
            self.tooltip = state.tooltip
            self._refresh_status_lines(state.menu)

        def _refresh_status_lines(self, lines: list[str]) -> None:
            for idx, item in enumerate(self._status_items):
                if idx < len(lines):
                    item.title = lines[idx]
                    item.show()
                else:
                    item.hide()

        def _animate(self, state, now: float) -> None:
            paths = self._icon_frames
            if state.icon_state == "alarm":
                icon = paths["dead"]
            elif state.icon_state == "running":
                if now - self._last_frame_at >= 1.0 / max(state.fps, 1.0):
                    self._frame_idx ^= 1
                    self._last_frame_at = now
                icon = paths["run1"] if self._frame_idx else paths["run2"]
            else:
                icon = paths["stand"]
            self.icon = str(icon)

    DinoApp().run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="anon-proxy menu-bar indicator")
    parser.add_argument("--url", default=None, help="status endpoint URL")
    parser.add_argument(
        "--watch", action="store_true", help="terminal status line instead of menu bar"
    )
    parser.add_argument(
        "--start-proxy", action="store_true", help="launch a supervised proxy on start"
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="inference backend for the supervised proxy (torch, onnx, auto)",
    )
    args = parser.parse_args(argv)

    url = args.url or cfg.load_config()["url"]
    if args.watch or sys.platform != "darwin":
        if sys.platform != "darwin" and not args.watch:
            print("menu bar is macOS-only; showing --watch terminal view instead.")
        watch_loop(url)
        return
    _run_macos_app(url, start_proxy=args.start_proxy, backend=args.backend)


if __name__ == "__main__":
    main()
