import os
import plistlib

from anon_proxy import app_bundle


def test_bundle_layout(tmp_path):
    app = app_bundle.build_app_bundle(
        tmp_path, exec_path="/usr/local/bin/anon-proxy-menubar"
    )
    stub = app / "Contents" / "MacOS" / "anon-proxy"
    plist = app / "Contents" / "Info.plist"

    assert app.name == "anon-proxy.app"
    assert stub.is_file()
    assert plist.is_file()
    assert os.access(stub, os.X_OK)
    assert "/usr/local/bin/anon-proxy-menubar" in stub.read_text()
    info = plistlib.loads(plist.read_bytes())
    assert info["LSUIElement"] is True
    assert info["CFBundleName"] == "anon-proxy"


def test_menubar_exec_path_falls_back(monkeypatch):
    monkeypatch.setattr(app_bundle.shutil, "which", lambda name: None)

    path = app_bundle.menubar_exec_path()

    assert "anon_proxy.menubar.app" in path
