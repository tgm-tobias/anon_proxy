from __future__ import annotations

import json

from anon_proxy.policy import Policy, mask_body
from tests.conftest import span


TEST_POLICY = Policy(
    pass_keys=frozenset(
        {"model", "role", "type", "id", "name", "tool_use_id", "media_type"}
    ),
    pass_paths=frozenset({("system",), ("tools",), ("metadata",)}),
    pass_block_types=frozenset({"thinking", "redacted_thinking"}),
    pass_block_subtrees={"image": "source"},
)


class TestFailClosedWalker:
    def test_unknown_field_is_masked(self, make_masker, fake_pipeline):
        masker = make_masker()
        fake_pipeline.set("Alice's data", [span("private_person", 0, 5, word="Alice")])
        body = {"model": "m", "some_future_field": "Alice's data"}

        out = mask_body(body, masker, TEST_POLICY)

        assert "Alice" not in json.dumps(out)

    def test_pass_keys_survive(self, make_masker):
        masker = make_masker()
        body = {
            "model": "claude-x",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }

        out = mask_body(body, masker, TEST_POLICY)

        assert out["model"] == "claude-x"
        assert out["messages"][0]["role"] == "user"

    def test_system_and_tools_subtrees_pass(self, make_masker, fake_pipeline):
        masker = make_masker()
        body = {
            "system": "You are X. Contact alice@x.com.",
            "tools": [{"name": "t", "description": "call alice@x.com"}],
        }

        out = mask_body(body, masker, TEST_POLICY)

        assert out == body

    def test_thinking_block_passes_whole(self, make_masker):
        masker = make_masker()
        block = {
            "type": "thinking",
            "thinking": "secret Alice reasoning",
            "signature": "sig==",
        }
        body = {"messages": [{"role": "assistant", "content": [block]}]}

        out = mask_body(body, masker, TEST_POLICY)

        assert out["messages"][0]["content"][0] == block

    def test_message_blocks_use_block_cache(self, make_masker, fake_pipeline):
        masker = make_masker()
        msg = {"role": "user", "content": "hello"}
        body = {"messages": [msg]}

        mask_body(body, masker, TEST_POLICY)
        calls_before = len(fake_pipeline.calls)
        mask_body(body, masker, TEST_POLICY)

        assert len(fake_pipeline.calls) == calls_before
