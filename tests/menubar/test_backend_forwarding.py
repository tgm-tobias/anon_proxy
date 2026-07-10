"""main() must forward --backend to the supervised proxy, not just to download."""

import anon_proxy.menubar.app as app


def test_main_forwards_backend_to_macos_app(monkeypatch):
    captured = {}

    def fake_run(url, *, start_proxy=False, backend=None):
        captured["start_proxy"] = start_proxy
        captured["backend"] = backend

    monkeypatch.setattr(app.sys, "platform", "darwin")
    monkeypatch.setattr(app, "_run_macos_app", fake_run)

    app.main(["--start-proxy", "--backend", "onnx"])

    assert captured["start_proxy"] is True
    assert captured["backend"] == "onnx"


def test_main_defaults_backend_to_none(monkeypatch):
    captured = {}

    def fake_run(url, *, start_proxy=False, backend=None):
        captured["backend"] = backend

    monkeypatch.setattr(app.sys, "platform", "darwin")
    monkeypatch.setattr(app, "_run_macos_app", fake_run)

    app.main([])

    assert captured["backend"] is None
