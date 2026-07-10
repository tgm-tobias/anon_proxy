import sys

from anon_proxy.menubar.supervisor import ProxySupervisor


def test_default_cmd_persists_store_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    sup = ProxySupervisor()
    cmd = sup._cmd

    assert cmd[:3] == [sys.executable, "-m", "anon_proxy.server"]
    assert "--store" in cmd
    assert "--metrics" in cmd
    assert cmd[cmd.index("--store") + 1].endswith("store.json")
    # No backend requested -> let the server pick its default (auto).
    assert "--backend" not in cmd


def test_backend_is_forwarded_into_default_cmd(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    sup = ProxySupervisor(backend="onnx")
    cmd = sup._cmd

    assert cmd[cmd.index("--backend") + 1] == "onnx"


def test_explicit_cmd_ignores_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    sup = ProxySupervisor(["custom", "cmd"], backend="onnx")

    assert sup._cmd == ["custom", "cmd"]
