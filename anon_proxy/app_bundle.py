"""Build a minimal macOS launcher app for the menu-bar entry point."""

from __future__ import annotations

import plistlib
import shutil
import sys
from pathlib import Path


def menubar_exec_path() -> str:
    found = shutil.which("anon-proxy-menubar")
    if found:
        return found
    return f"{sys.executable} -m anon_proxy.menubar.app"


def build_app_bundle(dest_dir: Path, *, exec_path: str) -> Path:
    app = dest_dir / "anon-proxy.app"
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)

    info = {
        "CFBundleName": "anon-proxy",
        "CFBundleDisplayName": "anon-proxy",
        "CFBundleIdentifier": "com.anon-proxy.menubar",
        "CFBundleExecutable": "anon-proxy",
        "CFBundlePackageType": "APPL",
        "LSUIElement": True,
    }
    (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps(info))

    stub = macos / "anon-proxy"
    stub.write_text(f'#!/bin/bash\nexec {exec_path} "$@"\n')
    stub.chmod(0o755)
    return app
