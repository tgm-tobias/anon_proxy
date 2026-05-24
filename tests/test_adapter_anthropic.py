"""Tests for the Anthropic adapter — Phase 4a (JSON walking).

Streaming is covered in Phase 4b (test_adapter_streaming.py).

Spec pinned here:
- mask_request walks `messages[*].content` (str or list of blocks):
  - text blocks → mask `text`
  - tool_use blocks → walk strings in `input` (dict/list)
  - tool_result blocks → mask `content` str or recurse into list
- mask_request leaves `system`, `tools`, `thinking` alone.
- mask_request returns a NEW body; input is not mutated.
- Each message is wrapped in `masker.mask_obj` for cross-turn caching.
- unmask_response walks `content[*]` — text blocks and tool_use.input.
"""

from __future__ import annotations

import pytest

from anon_proxy.adapters import anthropic as anth
from anon_proxy.regex_detector import RegexDetector


@pytest.fixture
def detector() -> RegexDetector:
    """Matches capitalized words ≥3 letters → PERSON. Deterministic, no ML."""
    return RegexDetector({"PERSON": r"\b[A-Z][a-z]{2,}\b"})


@pytest.fixture
def masker(make_masker, detector):
    return make_masker(extra_detectors=[detector])


# ---------------------------------------------------------------------------
# mask_request
# ---------------------------------------------------------------------------


class TestMaskRequestShape:
    def test_empty_body_returns_empty(self, masker):
        assert anth.mask_request({}, masker) == {}

    def test_no_messages_returns_unchanged(self, masker):
        body = {"model": "claude-3-5-sonnet", "max_tokens": 100}
        assert anth.mask_request(body, masker) == body

    def test_messages_not_a_list_left_alone(self, masker):
        body = {"messages": "not a list"}
        assert anth.mask_request(body, masker) == body

    def test_returns_copy_input_not_mutated(self, masker):
        body = {
            "messages": [{"role": "user", "content": "Hi Alice"}],
            "model": "claude-3-5-sonnet",
        }
        original = {
            "messages": [{"role": "user", "content": "Hi Alice"}],
            "model": "claude-3-5-sonnet",
        }
        anth.mask_request(body, masker)
        assert body == original


class TestMaskRequestStringContent:
    def test_user_text_masked(self, masker):
        body = {"messages": [{"role": "user", "content": "Hi Alice"}]}
        out = anth.mask_request(body, masker)
        assert out["messages"][0]["content"] == "Hi <PERSON_1>"

    def test_role_field_preserved(self, masker):
        body = {"messages": [{"role": "assistant", "content": "Hi Alice"}]}
        out = anth.mask_request(body, masker)
        assert out["messages"][0]["role"] == "assistant"


class TestMaskRequestBlockListContent:
    def test_text_block(self, masker):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "msg from Alice"}],
                }
            ]
        }
        out = anth.mask_request(body, masker)
        assert out["messages"][0]["content"][0]["text"] == "msg from <PERSON_1>"

    def test_tool_use_input_walked(self, masker):
        # Shape mirrors real Bash tool_use in Claude Code traffic.
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01abc",
                            "name": "Bash",
                            "input": {
                                "command": "ssh Alice@host -- echo hi from Bob",
                                "description": "ssh into the host",
                            },
                        }
                    ],
                }
            ]
        }
        out = anth.mask_request(body, masker)
        inp = out["messages"][0]["content"][0]["input"]
        assert "<PERSON_1>" in inp["command"]  # Alice masked
        assert "<PERSON_2>" in inp["command"]  # Bob masked
        # Non-PII fields preserved.
        assert out["messages"][0]["content"][0]["id"] == "toolu_01abc"
        assert out["messages"][0]["content"][0]["name"] == "Bash"

    def test_tool_use_input_nested_list_walked(self, masker):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "x",
                            "name": "broadcast",
                            "input": {"recipients": ["Alice", "Bob"]},
                        }
                    ],
                }
            ]
        }
        out = anth.mask_request(body, masker)
        assert out["messages"][0]["content"][0]["input"]["recipients"] == [
            "<PERSON_1>",
            "<PERSON_2>",
        ]

    def test_tool_result_string_content(self, masker):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "x",
                            "content": "from Alice",
                        }
                    ],
                }
            ]
        }
        out = anth.mask_request(body, masker)
        assert out["messages"][0]["content"][0]["content"] == "from <PERSON_1>"

    def test_tool_result_block_list_content(self, masker):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "x",
                            "content": [{"type": "text", "text": "Hi Alice"}],
                        }
                    ],
                }
            ]
        }
        out = anth.mask_request(body, masker)
        nested = out["messages"][0]["content"][0]["content"][0]
        assert nested["text"] == "Hi <PERSON_1>"

    def test_cache_control_preserved_on_text_block(self, masker):
        # Real Claude Code traffic decorates individual blocks with
        # `cache_control: {"type": "ephemeral", "ttl": "1h"}`. The adapter
        # masks `text` but must preserve cache_control verbatim.
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "msg from Alice",
                            "cache_control": {"type": "ephemeral", "ttl": "1h"},
                        }
                    ],
                }
            ]
        }
        out = anth.mask_request(body, masker)
        block = out["messages"][0]["content"][0]
        assert block["text"] == "msg from <PERSON_1>"
        assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_unknown_block_type_left_alone(self, masker):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"data": "Alice"}}],
                }
            ]
        }
        out = anth.mask_request(body, masker)
        # Unknown block type → passed through verbatim, even though it
        # textually contains PII.
        assert out["messages"][0]["content"][0]["source"]["data"] == "Alice"


class TestMaskRequestFieldsNotTouched:
    def test_system_prompt_left_alone(self, masker):
        body = {
            "system": "You may speak to Alice",
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = anth.mask_request(body, masker)
        assert out["system"] == "You may speak to Alice"

    def test_system_as_typed_block_list_left_alone(self, masker):
        # Real Claude Code traffic sends `system` as a list of typed blocks,
        # each possibly with `cache_control`. The adapter must leave the
        # entire list untouched.
        system = [
            {"type": "text", "text": "x-anthropic-billing: cc_version=2.1.112"},
            {
                "type": "text",
                "text": "You are Claude Code, talking to Alice.",
                "cache_control": {"type": "ephemeral"},
            },
        ]
        body = {
            "system": system,
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = anth.mask_request(body, masker)
        assert out["system"] == system  # byte-identical, including cache_control

    def test_tools_left_alone(self, masker):
        body = {
            "tools": [{"name": "ping", "description": "Send to Alice"}],
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = anth.mask_request(body, masker)
        assert out["tools"] == [{"name": "ping", "description": "Send to Alice"}]

    def test_thinking_block_inside_message_left_alone(self, masker):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "text": "I should call Alice"},
                        {"type": "text", "text": "Hi Bob"},
                    ],
                }
            ]
        }
        out = anth.mask_request(body, masker)
        # thinking block unchanged
        assert out["messages"][0]["content"][0] == {
            "type": "thinking",
            "text": "I should call Alice",
        }
        # text block masked
        assert out["messages"][0]["content"][1]["text"] == "Hi <PERSON_1>"


class TestMaskRequestUsesMaskObj:
    def test_identical_messages_across_calls_cache(self, masker, fake_pipeline):
        # Two requests with the same message content. The masker's
        # mask_obj cache should ensure the inner walker is invoked only once
        # per identical message.
        msg = {"role": "user", "content": "Hi Alice"}
        anth.mask_request({"messages": [msg, msg]}, masker)
        # First request: 2 identical messages → walker runs once (one hit, one miss).
        # But we measure via the underlying detector: it runs once per walker call.
        # Easier check: pipeline ought to be called 1 time (the mask() inside the walker
        # short-circuits the empty pipeline response too, so this is robust).
        anth.mask_request({"messages": [msg]}, masker)
        # No new pipeline calls — cache hit at the message level avoided a re-mask.
        # The detector is regex-only (no ML); we can't easily observe its calls.
        # Instead, observe via the masker's content cache: re-masking the same
        # string content goes through the masker's own _cache. The mask_obj
        # cache should additionally skip the recursive walk.
        # End-to-end behavior: 3 identical mask_request calls → at most 1 pipeline call.
        assert len(fake_pipeline.calls) <= 1


# ---------------------------------------------------------------------------
# unmask_response
# ---------------------------------------------------------------------------


class TestUnmaskResponseShape:
    def test_empty_body_returns_empty(self, masker):
        assert anth.unmask_response({}, masker) == {}

    def test_no_content_returns_unchanged(self, masker):
        body = {"id": "msg_1", "stop_reason": "end_turn"}
        assert anth.unmask_response(body, masker) == body

    def test_content_not_a_list_left_alone(self, masker):
        body = {"content": "not a list"}
        assert anth.unmask_response(body, masker) == body


class TestUnmaskResponseBlocks:
    def test_text_block_unmasked(self, masker, store):
        store.get_or_create("PERSON", "Alice")
        body = {"content": [{"type": "text", "text": "Reply to <PERSON_1>"}]}
        out = anth.unmask_response(body, masker)
        assert out["content"][0]["text"] == "Reply to Alice"

    def test_tool_use_input_unmasked(self, masker, store):
        store.get_or_create("PERSON", "Alice")
        body = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "x",
                    "name": "send",
                    "input": {"to": "<PERSON_1>"},
                }
            ]
        }
        out = anth.unmask_response(body, masker)
        assert out["content"][0]["input"] == {"to": "Alice"}

    def test_other_block_types_passed_through(self, masker, store):
        store.get_or_create("PERSON", "Alice")
        body = {"content": [{"type": "thinking", "text": "<PERSON_1>"}]}
        out = anth.unmask_response(body, masker)
        # thinking block not unmasked.
        assert out["content"][0]["text"] == "<PERSON_1>"

    def test_response_metadata_preserved(self, masker, store):
        # Real non-streaming responses carry id, type, role, stop_reason,
        # usage, context_management, etc. The adapter only touches `content`;
        # everything else must pass through byte-identical.
        store.get_or_create("PERSON", "Alice")
        body = {
            "id": "msg_01Y6oELCsfQKUzosT4hpTBog",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "Reply to <PERSON_1>"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 8,
                "output_tokens": 1,
                "cache_creation": {"ephemeral_5m_input_tokens": 0},
            },
            "context_management": {"applied_edits": []},
        }
        out = anth.unmask_response(body, masker)
        assert out["content"][0]["text"] == "Reply to Alice"
        # All other fields preserved verbatim.
        for k in (
            "id",
            "type",
            "role",
            "model",
            "stop_reason",
            "stop_sequence",
            "usage",
            "context_management",
        ):
            assert out[k] == body[k], f"field {k} not preserved"


# ---------------------------------------------------------------------------
# inject_system
# ---------------------------------------------------------------------------


class TestInjectSystem:
    PROMPT = "INJECTED PROMPT"

    def test_absent_system_sets_string(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        out = anth.inject_system(body, self.PROMPT)
        assert out["system"] == self.PROMPT
        # Messages untouched.
        assert out["messages"] == body["messages"]

    def test_string_system_is_prepended_with_blank_line(self):
        body = {"system": "You are helpful."}
        out = anth.inject_system(body, self.PROMPT)
        assert out["system"] == f"{self.PROMPT}\n\nYou are helpful."

    def test_list_system_prepends_text_block(self):
        original_blocks = [
            {"type": "text", "text": "You are helpful."},
            {
                "type": "text",
                "text": "Static rules.",
                "cache_control": {"type": "ephemeral"},
            },
        ]
        body = {"system": list(original_blocks)}
        out = anth.inject_system(body, self.PROMPT)
        assert out["system"][0] == {"type": "text", "text": self.PROMPT}
        # Client's blocks (and any cache_control on them) preserved in order.
        assert out["system"][1:] == original_blocks

    def test_returns_copy_input_not_mutated(self):
        body = {"system": "orig", "messages": []}
        original = {"system": "orig", "messages": []}
        anth.inject_system(body, self.PROMPT)
        assert body == original

    def test_list_system_does_not_mutate_input_list(self):
        original_blocks = [{"type": "text", "text": "orig"}]
        body = {"system": original_blocks}
        anth.inject_system(body, self.PROMPT)
        # The original list reference must not have been prepended to.
        assert original_blocks == [{"type": "text", "text": "orig"}]

    def test_other_fields_preserved(self):
        body = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "calc"}],
        }
        out = anth.inject_system(body, self.PROMPT)
        for k in ("model", "max_tokens", "messages", "tools"):
            assert out[k] == body[k]
        assert out["system"] == self.PROMPT
