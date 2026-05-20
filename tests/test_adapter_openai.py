"""Tests for the OpenAI adapter — Phase 4a (JSON walking).

Streaming is covered in Phase 4b (test_adapter_streaming.py).

Spec pinned here:
- mask_request walks:
  - messages[*].content (str OR list of items; text items only; image_url skipped)
  - messages[*].tool_calls[*].function.arguments (JSON-string or dict)
  - tools[*].function.parameters (schema)
- Tool descriptions (tools[*].function.description) are NOT masked.
- Each message wrapped in masker.mask_obj for cross-turn caching (matches Anthropic).
- mask_request returns a NEW body; input not mutated.
- unmask_response walks choices[*].message.content and tool_calls arguments.
"""

from __future__ import annotations

import json

import pytest

from anon_proxy.adapters import openai as oai
from anon_proxy.regex_detector import RegexDetector


@pytest.fixture
def detector() -> RegexDetector:
    return RegexDetector({"PERSON": r"\b[A-Z][a-z]{2,}\b"})


@pytest.fixture
def masker(make_masker, detector):
    return make_masker(extra_detectors=[detector])


# ---------------------------------------------------------------------------
# mask_request
# ---------------------------------------------------------------------------


class TestMaskRequestShape:
    def test_empty_body(self, masker):
        assert oai.mask_request({}, masker) == {}

    def test_no_messages(self, masker):
        body = {"model": "gpt-4"}
        assert oai.mask_request(body, masker) == body

    def test_returns_copy_input_not_mutated(self, masker):
        body = {"messages": [{"role": "user", "content": "Hi Alice"}]}
        original = {"messages": [{"role": "user", "content": "Hi Alice"}]}
        oai.mask_request(body, masker)
        assert body == original


class TestMaskRequestStringContent:
    def test_user_content_masked(self, masker):
        body = {"messages": [{"role": "user", "content": "Hi Alice"}]}
        out = oai.mask_request(body, masker)
        assert out["messages"][0]["content"] == "Hi <PERSON_1>"


class TestMaskRequestArrayContent:
    def test_text_item_masked(self, masker):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi Alice"}],
                }
            ]
        }
        out = oai.mask_request(body, masker)
        assert out["messages"][0]["content"][0]["text"] == "Hi <PERSON_1>"

    def test_image_url_left_alone(self, masker):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hi Alice"},
                        {"type": "image_url", "image_url": {"url": "https://x/Alice.png"}},
                    ],
                }
            ]
        }
        out = oai.mask_request(body, masker)
        assert out["messages"][0]["content"][0]["text"] == "Hi <PERSON_1>"
        # Image url unchanged.
        assert out["messages"][0]["content"][1]["image_url"]["url"] == "https://x/Alice.png"


class TestMaskRequestToolCalls:
    def test_function_arguments_json_string_parsed_walked_redumped(self, masker):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {
                                "name": "send",
                                "arguments": json.dumps({"to": "Alice", "body": "Hi"}),
                            },
                        }
                    ],
                }
            ]
        }
        out = oai.mask_request(body, masker)
        args = json.loads(out["messages"][0]["tool_calls"][0]["function"]["arguments"])
        assert args == {"to": "<PERSON_1>", "body": "Hi"}

    def test_function_arguments_invalid_json_masked_as_string(self, masker):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "send", "arguments": "not json: Alice"},
                        }
                    ],
                }
            ]
        }
        out = oai.mask_request(body, masker)
        # JSONDecodeError fallback: arguments masked as a raw string.
        assert (
            out["messages"][0]["tool_calls"][0]["function"]["arguments"]
            == "not json: <PERSON_1>"
        )

    def test_function_arguments_dict_walked(self, masker):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "send", "arguments": {"to": "Alice"}},
                        }
                    ],
                }
            ]
        }
        out = oai.mask_request(body, masker)
        assert out["messages"][0]["tool_calls"][0]["function"]["arguments"] == {
            "to": "<PERSON_1>"
        }


class TestMaskRequestTools:
    def test_description_NOT_masked(self, masker):
        # Phase 4a redesign: tool descriptions are static schema; do not mask.
        body = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "send",
                        "description": "Sends a message to Alice",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = oai.mask_request(body, masker)
        assert out["tools"][0]["function"]["description"] == "Sends a message to Alice"

    def test_parameters_schema_walked(self, masker):
        body = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "send",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "to": {"type": "string", "description": "name like Alice"}
                            },
                        },
                    },
                }
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = oai.mask_request(body, masker)
        desc = out["tools"][0]["function"]["parameters"]["properties"]["to"]["description"]
        assert desc == "name like <PERSON_1>"


class TestMaskRequestUsesMaskObj:
    def test_identical_messages_share_cache(self, masker, fake_pipeline):
        msg = {"role": "user", "content": "Hi Alice"}
        oai.mask_request({"messages": [msg, msg]}, masker)
        oai.mask_request({"messages": [msg]}, masker)
        # Same observability as Anthropic: identical message content → cache hits
        # → pipeline called at most once across all three identical message walks.
        assert len(fake_pipeline.calls) <= 1


# ---------------------------------------------------------------------------
# unmask_response
# ---------------------------------------------------------------------------


class TestUnmaskResponseShape:
    def test_empty_body(self, masker):
        assert oai.unmask_response({}, masker) == {}

    def test_no_choices(self, masker):
        body = {"id": "x"}
        assert oai.unmask_response(body, masker) == body


class TestUnmaskResponseContent:
    def test_string_content_unmasked(self, masker, store):
        store.get_or_create("PERSON", "Alice")
        body = {
            "choices": [{"message": {"role": "assistant", "content": "Hi <PERSON_1>"}}]
        }
        out = oai.unmask_response(body, masker)
        assert out["choices"][0]["message"]["content"] == "Hi Alice"

    def test_array_content_text_unmasked(self, masker, store):
        store.get_or_create("PERSON", "Alice")
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hi <PERSON_1>"}],
                    }
                }
            ]
        }
        out = oai.unmask_response(body, masker)
        assert out["choices"][0]["message"]["content"][0]["text"] == "Hi Alice"


class TestUnmaskResponseToolCalls:
    def test_arguments_json_string_unmasked(self, masker, store):
        store.get_or_create("PERSON", "Alice")
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "x",
                                "type": "function",
                                "function": {
                                    "name": "send",
                                    "arguments": json.dumps({"to": "<PERSON_1>"}),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        out = oai.unmask_response(body, masker)
        args = json.loads(out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        assert args == {"to": "Alice"}
