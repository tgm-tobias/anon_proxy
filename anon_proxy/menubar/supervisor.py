"""Proxy subprocess lifecycle plus a launchd Start-at-login agent."""

from __future__ import annotations

import atexit
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

_PLIST_TMPL = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>
{args}
  </array>
  <key>RunAtLoad</key><{run_at_load}/>
</dict>
</plist>
"""


class ProxySupervisor:
    def __init__(self, cmd: list[str] | None = None) -> None:
        self._cmd = cmd or [sys.executable, "-m", "anon_proxy.server"]
        self._proc: subprocess.Popen | None = None
        atexit.register(self.stop)

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, extra_args: list[str] | None = None) -> None:
        if self.is_running():
            return
        self._proc = subprocess.Popen(self._cmd + list(extra_args or []))

    def stop(self, grace: float = 5.0) -> None:
        if not self.is_running():
            self._proc = None
            return
        assert self._proc is not None
        self._proc.terminate()
        try:
            self._proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    def restart(self, extra_args: list[str] | None = None) -> None:
        self.stop()
        self.start(extra_args)


def launch_agent_plist(
    label: str, program_args: list[str], *, run_at_load: bool = True
) -> str:
    args = "\n".join(f"    <string>{escape(arg)}</string>" for arg in program_args)
    return _PLIST_TMPL.format(
        label=escape(label),
        args=args,
        run_at_load="true" if run_at_load else "false",
    )


def _plist_path(label: str, plist_dir: Path | None) -> Path:
    base = (
        plist_dir if plist_dir is not None else Path.home() / "Library" / "LaunchAgents"
    )
    return base / f"{label}.plist"


def install_launch_agent(
    label: str,
    program_args: list[str],
    *,
    plist_dir: Path | None = None,
    load: bool = True,
) -> Path:
    path = _plist_path(label, plist_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(launch_agent_plist(label, program_args))
    if load:
        subprocess.run(["launchctl", "load", str(path)], check=False)
    return path


def uninstall_launch_agent(
    label: str, *, plist_dir: Path | None = None, load: bool = True
) -> None:
    path = _plist_path(label, plist_dir)
    if load and path.exists():
        subprocess.run(["launchctl", "unload", str(path)], check=False)
    path.unlink(missing_ok=True)
