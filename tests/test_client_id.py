import pytest

from anon_proxy.client_id import classify_client


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (
            {"user-agent": "claude-cli/1.2.3 (external, cli)", "x-app": "cli"},
            "Claude Code",
        ),
        ({"user-agent": "codex_cli_rs/0.4.0"}, "Codex"),
        ({"originator": "codex_cli_rs", "user-agent": "reqwest/0.12"}, "Codex"),
        (
            {
                "user-agent": "Anthropic/Python 0.96.0",
                "x-stainless-lang": "python",
                "x-stainless-package-version": "0.96.0",
                "anthropic-version": "2023-06-01",
            },
            "Anthropic SDK",
        ),
        (
            {
                "user-agent": "OpenAI/Python 1.40.0",
                "x-stainless-lang": "python",
                "x-stainless-package-version": "1.40.0",
            },
            "OpenAI SDK",
        ),
        ({"user-agent": "curl/8.4.0"}, "curl"),
        ({}, "unknown"),
    ],
)
def test_classify_client(headers, expected):
    assert classify_client(headers) == expected
