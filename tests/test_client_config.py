from anon_proxy import client_config


def test_base_url_includes_provider_path():
    assert client_config.base_url_for("claude") == "http://127.0.0.1:8080/anthropic"
    assert (
        client_config.base_url_for("openai", port=9000)
        == "http://127.0.0.1:9000/openai"
    )


def test_env_snippet_uses_right_var():
    url = client_config.base_url_for("claude")
    assert client_config.env_snippet("claude", url) == (
        "export ANTHROPIC_BASE_URL=http://127.0.0.1:8080/anthropic"
    )


def test_apply_env_is_idempotent(tmp_path):
    rc_path = tmp_path / ".zshrc"
    rc_path.write_text("# existing\n")
    url = client_config.base_url_for("claude")

    client_config.apply_env("claude", url, rc_path)
    client_config.apply_env("claude", url, rc_path)

    body = rc_path.read_text()
    assert body.count("export ANTHROPIC_BASE_URL=") == 1
    assert body.startswith("# existing")
