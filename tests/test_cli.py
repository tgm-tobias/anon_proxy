from anon_proxy import cli


def test_bare_flags_route_to_serve(monkeypatch):
    seen = {}

    def fake_serve(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(cli, "_serve", fake_serve)

    assert cli.main(["--port", "9000"]) == 0

    assert seen["argv"] == ["--port", "9000"]


def test_serve_subcommand_strips_the_verb(monkeypatch):
    seen = {}

    def fake_serve(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(cli, "_serve", fake_serve)

    assert cli.main(["serve", "--debug"]) == 0

    assert seen["argv"] == ["--debug"]


def test_known_verbs_registered():
    assert set(cli._COMMANDS) == {"download-model", "setup-client", "install-app", "up"}


def test_up_prefetches_when_uncached(monkeypatch):
    events = []
    monkeypatch.setattr("anon_proxy.model_cache.is_cached", lambda backend: False)
    monkeypatch.setattr(
        "anon_proxy.model_cache.download_model",
        lambda backend, progress=True: events.append(("download", backend)) or "/dir",
    )
    monkeypatch.setattr(
        "anon_proxy.menubar.app.main", lambda argv=None: events.append(("gui", argv))
    )

    assert cli.main(["up", "--backend", "onnx-q4f16"]) == 0

    assert ("download", "onnx-q4f16") in events
    gui_calls = [argv for tag, argv in events if tag == "gui"]
    assert gui_calls, "GUI path was not launched"
    gui_argv = gui_calls[0]
    assert "--start-proxy" in gui_argv
    # The backend the user requested must reach the supervised proxy, not just
    # the download step — the normalized runtime name (onnx-q4f16 -> onnx).
    assert gui_argv[gui_argv.index("--backend") + 1] == "onnx"


def test_up_skips_download_when_cached(monkeypatch):
    events = []
    monkeypatch.setattr("anon_proxy.model_cache.is_cached", lambda backend: True)
    monkeypatch.setattr(
        "anon_proxy.model_cache.download_model",
        lambda *args, **kwargs: events.append("download"),
    )
    monkeypatch.setattr(
        "anon_proxy.menubar.app.main", lambda argv=None: events.append("gui")
    )

    assert cli.main(["up"]) == 0

    assert "download" not in events
    assert "gui" in events


def test_up_no_gui_starts_stateful_server(monkeypatch, tmp_path):
    events = []
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setattr("anon_proxy.model_cache.is_cached", lambda backend: True)
    monkeypatch.setattr(cli, "_serve", lambda argv: events.append(("serve", argv)) or 0)

    assert cli.main(["up", "--backend", "onnx-q4f16", "--no-gui"]) == 0

    argv = events[0][1]
    assert "--store" in argv
    assert argv[argv.index("--store") + 1].endswith("store.json")
    assert argv[argv.index("--backend") + 1] == "onnx"
