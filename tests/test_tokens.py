from anon_proxy.tokens import approx_tokens_from_text, extract_output_tokens


def test_approx_tokens():
    assert approx_tokens_from_text("") == 0
    assert approx_tokens_from_text("a") == 1
    assert approx_tokens_from_text("x" * 40) == 10


def test_extract_anthropic_output_tokens():
    resp = {"usage": {"input_tokens": 5, "output_tokens": 42}}
    assert extract_output_tokens("anthropic", resp) == 42


def test_extract_openai_output_tokens():
    resp = {"usage": {"prompt_tokens": 5, "completion_tokens": 17}}
    assert extract_output_tokens("openai", resp) == 17


def test_extract_returns_none_when_absent():
    assert extract_output_tokens("anthropic", {}) is None
    assert extract_output_tokens("openai", {"usage": {}}) is None
    assert extract_output_tokens("anthropic", {"usage": "nope"}) is None
