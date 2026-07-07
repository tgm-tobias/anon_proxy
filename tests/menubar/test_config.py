from anon_proxy.menubar import config


def test_load_returns_defaults_when_missing(tmp_path):
    cfg = config.load_config(tmp_path / "nope.json")
    assert cfg == config.DEFAULTS
    assert cfg is not config.DEFAULTS


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "sub" / "menubar.json"
    config.save_config(
        {
            "theme": "winter",
            "start_at_login": True,
            "url": "http://x/_status",
        },
        path,
    )
    cfg = config.load_config(path)
    assert cfg["theme"] == "winter"
    assert cfg["start_at_login"] is True


def test_partial_file_is_merged_over_defaults(tmp_path):
    path = tmp_path / "menubar.json"
    path.write_text('{"theme": "halloween"}')
    cfg = config.load_config(path)
    assert cfg["theme"] == "halloween"
    assert cfg["start_at_login"] is False


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "menubar.json"
    path.write_text("{not json")
    assert config.load_config(path) == config.DEFAULTS
