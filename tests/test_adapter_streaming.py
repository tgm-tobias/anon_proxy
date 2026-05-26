"""Tests for SSE streaming in both adapters — Phase 4b.

Agreed unified flush rule (matches existing Anthropic _split_emit):
- Buffer each block's incoming delta content.
- Emit everything up to the last unterminated `<` (a potentially-incomplete
  placeholder); hold the rest for the next chunk.
- At block end (content_block_stop / [DONE] / stream EOF), unmask and flush
  the remaining buffer as a final delta event.
- No size cap — placeholders are bounded by label length plus `_<digits>`.

Out of scope (pinned current behavior):
- OpenAI multi-choice (`n>1`) streaming: only the first choice is processed.
- `tool_call_buffers` default-to-index-0 when index is missing.
- Anthropic trailing-bytes passthrough on truncated upstream (no terminator).

Known limitations (documented via test):
- PII whose original value contains a literal `<` will cause the next chunk
  to hold its tail until a `>` appears. Latency, not correctness.
"""

from __future__ import annotations

import json

import pytest

from anon_proxy.adapters import anthropic as anth
from anon_proxy.adapters import openai as oai


async def _aiter(chunks):
    """Wrap a list of bytes into an async iterator."""
    for c in chunks:
        yield c


async def _collect(stream) -> bytes:
    """Consume an async byte iterator into a single bytes blob."""
    parts = []
    async for piece in stream:
        parts.append(piece)
    return b"".join(parts)


def _reconstruct_text_deltas(out: bytes) -> str:
    """Concatenate every Anthropic text_delta `text` field in event order."""
    decoded = ""
    for line in out.decode().split("\n"):
        if not line.startswith("data:"):
            continue
        try:
            d = json.loads(line[len("data:") :].strip())
        except json.JSONDecodeError:
            continue
        delta = d.get("delta", {})
        if delta.get("type") == "text_delta":
            decoded += delta.get("text", "")
    return decoded


def _reconstruct_partial_json(out: bytes) -> str:
    """Concatenate every Anthropic input_json_delta `partial_json` field."""
    decoded = ""
    for line in out.decode().split("\n"):
        if not line.startswith("data:"):
            continue
        try:
            d = json.loads(line[len("data:") :].strip())
        except json.JSONDecodeError:
            continue
        delta = d.get("delta", {})
        if delta.get("type") == "input_json_delta":
            decoded += delta.get("partial_json", "")
    return decoded


def _reconstruct_openai_content(out: bytes) -> str:
    """Concatenate every OpenAI delta.content across events (first choice)."""
    decoded = ""
    for line in out.decode().split("\n"):
        if not line.startswith("data:") or "[DONE]" in line:
            continue
        try:
            d = json.loads(line[len("data:") :].strip())
        except json.JSONDecodeError:
            continue
        choices = d.get("choices", [])
        if not choices:
            continue
        c = choices[0].get("delta", {}).get("content")
        if isinstance(c, str):
            decoded += c
    return decoded


def _make_masker_with_tokens(make_filter, store, *pairs):
    from anon_proxy.masker import Masker

    m = Masker(filter=make_filter(), store=store, skip_patterns=[])
    for label, value in pairs:
        store.get_or_create(label, value)
    return m


# ---------------------------------------------------------------------------
# Anthropic — _split_emit helper (the canonical rule everyone aligns to)
# ---------------------------------------------------------------------------


class TestSplitEmit:
    def test_no_open_bracket_emits_all(self):
        from anon_proxy.adapters.anthropic import _split_emit

        assert _split_emit("plain text") == ("plain text", "")

    def test_complete_token_emits_all(self):
        from anon_proxy.adapters.anthropic import _split_emit

        assert _split_emit("hi <PERSON_1> bye") == ("hi <PERSON_1> bye", "")

    def test_incomplete_token_holds_tail(self):
        from anon_proxy.adapters.anthropic import _split_emit

        # Last `<` has no `>` after it — hold from that `<` onward.
        assert _split_emit("hi <PER") == ("hi ", "<PER")

    def test_multiple_brackets_only_last_unterminated_matters(self):
        from anon_proxy.adapters.anthropic import _split_emit

        # Earlier `<>` pairs are fine; only the trailing unterminated one is held.
        assert _split_emit("<PERSON_1> then <PE") == ("<PERSON_1> then ", "<PE")


# ---------------------------------------------------------------------------
# Anthropic streaming — content_block_delta flow
# ---------------------------------------------------------------------------


def _sse_event(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


@pytest.mark.asyncio
class TestAnthropicTextDelta:
    async def test_placeholder_in_single_delta_unmasked(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        chunks = [
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hi <PERSON_1>"},
                },
            ),
            _sse_event(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": 0,
                },
            ),
        ]
        out = (await _collect(anth.transform_stream(_aiter(chunks), m))).decode()
        assert "Hi Alice" in out
        assert "<PERSON_1>" not in out

    async def test_placeholder_split_across_two_deltas(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        chunks = [
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hi <PER"},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "SON_1> there"},
                },
            ),
            _sse_event(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": 0,
                },
            ),
        ]
        out = await _collect(anth.transform_stream(_aiter(chunks), m))
        # Reconstruct the client's view: concatenate text from each text_delta.
        assert _reconstruct_text_deltas(out) == "Hi Alice there"
        # Neither partial nor whole placeholder leaks to any individual delta.
        decoded = out.decode()
        assert "<PER" not in decoded
        assert "<PERSON_1>" not in decoded

    async def test_placeholder_split_char_by_char(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        text = "Hi <PERSON_1> there"
        chunks = [
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        ]
        chunks += [
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": ch},
                },
            )
            for ch in text
        ]
        chunks.append(
            _sse_event(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": 0,
                },
            )
        )
        out = await _collect(anth.transform_stream(_aiter(chunks), m))
        assert _reconstruct_text_deltas(out) == "Hi Alice there"


@pytest.mark.asyncio
class TestAnthropicBufferFlushOnBlockStop:
    async def test_incomplete_placeholder_flushed_on_stop(self, make_filter, store):
        """If the stream ends with an incomplete placeholder still in the
        buffer, content_block_stop flushes it (the placeholder regex won't
        match, so it passes through as-is — but no data is lost)."""
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        chunks = [
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "trailing <inc"},
                },
            ),
            _sse_event(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": 0,
                },
            ),
        ]
        out = (await _collect(anth.transform_stream(_aiter(chunks), m))).decode()
        # The held tail "<inc" is flushed verbatim (not a real placeholder).
        assert "<inc" in out

    async def test_complete_placeholder_at_stop_unmasked(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        chunks = [
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hi <PER"},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "SON_1>"},
                },
            ),
            _sse_event(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": 0,
                },
            ),
        ]
        out = await _collect(anth.transform_stream(_aiter(chunks), m))
        assert _reconstruct_text_deltas(out) == "hi Alice"


@pytest.mark.asyncio
class TestAnthropicThinkingDelta:
    async def test_thinking_delta_unmasked(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        chunks = [
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "sig",
                    },
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": "thinking about <PERSON_1>",
                    },
                },
            ),
            _sse_event(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": 0,
                },
            ),
        ]
        out = (await _collect(anth.transform_stream(_aiter(chunks), m))).decode()
        assert "thinking about Alice" in out
        assert "<PERSON_1>" not in out

    async def test_thinking_delta_split_across_chunks(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        chunks = [
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "hi <PER"},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "SON_1>"},
                },
            ),
            _sse_event(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": 0,
                },
            ),
        ]
        out = await _collect(anth.transform_stream(_aiter(chunks), m))
        decoded = out.decode()
        # "hi " is emitted in the first delta, "Alice" in the second — they
        # are never combined into a single "hi Alice" in the raw SSE text.
        assert '"thinking": "hi "' in decoded
        assert '"thinking": "Alice"' in decoded
        assert "<PER" not in decoded
        assert "<PERSON_1>" not in decoded


@pytest.mark.asyncio
class TestAnthropicMultipleBlocks:
    async def test_independent_blocks_have_independent_buffers(
        self, make_filter, store
    ):
        m = _make_masker_with_tokens(
            make_filter, store, ("PERSON", "Alice"), ("PERSON", "Bob")
        )
        chunks = [
            # Block 0 starts and emits incomplete placeholder
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "<PER"},
                },
            ),
            # Block 1 starts (independent buffer)
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "<PERSON_2>"},
                },
            ),
            _sse_event(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": 1,
                },
            ),
            # Block 0 completes
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "SON_1>"},
                },
            ),
            _sse_event(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": 0,
                },
            ),
        ]
        out = await _collect(anth.transform_stream(_aiter(chunks), m))
        # Per-block reconstruction: index 0 → "Alice", index 1 → "Bob".
        per_block: dict[int, str] = {}
        for line in out.decode().split("\n"):
            if not line.startswith("data:"):
                continue
            try:
                d = json.loads(line[len("data:") :].strip())
            except json.JSONDecodeError:
                continue
            delta = d.get("delta", {})
            if delta.get("type") == "text_delta":
                per_block[d.get("index")] = per_block.get(
                    d.get("index"), ""
                ) + delta.get("text", "")
        assert per_block.get(0) == "Alice"
        assert per_block.get(1) == "Bob"
        decoded = out.decode()
        assert "<PER" not in decoded
        assert "<PERSON_2>" not in decoded


@pytest.mark.asyncio
class TestAnthropicPassthrough:
    async def test_ping_event_unchanged(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store)
        chunk = b'event: ping\ndata: {"type": "ping"}\n\n'
        out = await _collect(anth.transform_stream(_aiter([chunk]), m))
        assert b"event: ping" in out
        assert b'"type": "ping"' in out

    async def test_unknown_event_passed_through(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store)
        chunk = _sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 10},
            },
        )
        out = (await _collect(anth.transform_stream(_aiter([chunk]), m))).decode()
        assert "message_delta" in out
        assert '"stop_reason": "end_turn"' in out or '"stop_reason":"end_turn"' in out

    async def test_trailing_bytes_without_terminator_passed_through(
        self, make_filter, store
    ):
        # Per the agreed scope, trailing bytes are emitted as-is (best-effort).
        m = _make_masker_with_tokens(make_filter, store)
        chunks = [b"raw trailing bytes with no terminator"]
        out = await _collect(anth.transform_stream(_aiter(chunks), m))
        assert out == b"raw trailing bytes with no terminator"


@pytest.mark.asyncio
class TestAnthropicToolUseJsonDelta:
    async def test_input_json_delta_uses_unmask_json(self, make_filter, store):
        # PII original contains a backslash; in JSON-string context it must be
        # JSON-escaped so the surrounding JSON stays valid when re-parsed.
        m = _make_masker_with_tokens(make_filter, store, ("PATH", r"C:\users"))
        chunks = [
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "x",
                        "name": "t",
                        "input": {},
                    },
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"p":"<PATH_1>"}',
                    },
                },
            ),
            _sse_event(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": 0,
                },
            ),
        ]
        out = await _collect(anth.transform_stream(_aiter(chunks), m))
        # Reconstruct partial_json fields, parse as JSON, verify the unescaped
        # value is the original `C:\users` (a single backslash).
        accum = _reconstruct_partial_json(out)
        parsed = json.loads(accum)
        assert parsed == {"p": r"C:\users"}


@pytest.mark.asyncio
class TestAnthropicLiteralOpenAngleInPII:
    """Documented limitation: if PII original contains `<`, the unmasked
    output's `<` causes the *next* chunk's split_emit to hold its tail.
    Single-chunk behavior is correct; only multi-chunk affected.
    """

    async def test_single_chunk_with_angle_pii_correct(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store, ("NAME", "<unknown>"))
        chunks = [
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "<NAME_1>"},
                },
            ),
            _sse_event(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": 0,
                },
            ),
        ]
        out = (await _collect(anth.transform_stream(_aiter(chunks), m))).decode()
        assert "<unknown>" in out


# ---------------------------------------------------------------------------
# OpenAI streaming — agreed redesign target.
# ---------------------------------------------------------------------------


def _oai_event(data: dict | str) -> bytes:
    if isinstance(data, str):
        return f"data: {data}\n\n".encode()
    return f"data: {json.dumps(data)}\n\n".encode()


@pytest.mark.asyncio
class TestOpenAIContentDelta:
    async def test_placeholder_in_single_delta_unmasked(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        chunks = [
            _oai_event({"choices": [{"delta": {"content": "Hi <PERSON_1>"}}]}),
            _oai_event("[DONE]"),
        ]
        out = (await _collect(oai.transform_stream(_aiter(chunks), m))).decode()
        assert "Hi Alice" in out
        assert "<PERSON_1>" not in out

    async def test_placeholder_split_across_two_deltas(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        chunks = [
            _oai_event({"choices": [{"delta": {"content": "Hi <PER"}}]}),
            _oai_event({"choices": [{"delta": {"content": "SON_1> there"}}]}),
            _oai_event("[DONE]"),
        ]
        out = await _collect(oai.transform_stream(_aiter(chunks), m))
        assert _reconstruct_openai_content(out) == "Hi Alice there"

    async def test_partial_placeholder_never_leaks(self, make_filter, store):
        """The previous bug: 500-char fallback flushed the buffer with
        placeholders intact. The unified rule must never emit `<PER` or
        `<PERSON_1>` to the wire."""
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        chunks = [
            _oai_event({"choices": [{"delta": {"content": "Hi <PER"}}]}),
            _oai_event({"choices": [{"delta": {"content": "SON_1>"}}]}),
            _oai_event("[DONE]"),
        ]
        out = await _collect(oai.transform_stream(_aiter(chunks), m))
        # The reconstructed client view must equal the unmasked original AND
        # no individual delta's `content` field may contain a placeholder.
        assert _reconstruct_openai_content(out) == "Hi Alice"
        for line in out.decode().split("\n"):
            if line.startswith("data:") and "[DONE]" not in line:
                try:
                    d = json.loads(line[len("data:") :].strip())
                except json.JSONDecodeError:
                    continue
                c = d.get("choices", [{}])[0].get("delta", {}).get("content")
                if isinstance(c, str):
                    assert "<PER" not in c
                    assert "<PERSON_1>" not in c

    async def test_done_flushes_buffer(self, make_filter, store):
        """If the stream ends before unmasking can resolve, [DONE] must
        flush the buffer rather than leaving content held."""
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        chunks = [
            _oai_event({"choices": [{"delta": {"content": "trailing <PER"}}]}),
            _oai_event({"choices": [{"delta": {"content": "SON_1>"}}]}),
            _oai_event("[DONE]"),
        ]
        out = await _collect(oai.transform_stream(_aiter(chunks), m))
        assert _reconstruct_openai_content(out) == "trailing Alice"

    async def test_large_buffer_with_unresolved_placeholder_never_leaks(
        self, make_filter, store
    ):
        """Exposes the 500-char fallback bug: stream a long unresolved
        placeholder tail and verify no individual delta's `content` ever
        carries a `<` prefix to the wire. The unified rule should hold the
        tail indefinitely (or until block end), regardless of buffer size.
        """
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        # 600 chars of plain prefix, then an unresolved placeholder tail.
        prefix = "x" * 600
        chunks = [
            _oai_event({"choices": [{"delta": {"content": prefix + "<PER"}}]}),
            _oai_event({"choices": [{"delta": {"content": "SON_1>"}}]}),
            _oai_event("[DONE]"),
        ]
        out = await _collect(oai.transform_stream(_aiter(chunks), m))
        # Reconstructed view: the prefix + unmasked name, exactly.
        assert _reconstruct_openai_content(out) == prefix + "Alice"
        # No partial placeholder leaked.
        for line in out.decode().split("\n"):
            if line.startswith("data:") and "[DONE]" not in line:
                try:
                    d = json.loads(line[len("data:") :].strip())
                except json.JSONDecodeError:
                    continue
                c = d.get("choices", [{}])[0].get("delta", {}).get("content")
                if isinstance(c, str):
                    assert "<PER" not in c, f"partial placeholder leaked: {c!r}"


@pytest.mark.asyncio
class TestOpenAIDocumentedLimitations:
    """Pinned-current behaviors that we deliberately did NOT fix in 4b."""

    async def test_only_first_choice_processed(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        chunks = [
            _oai_event(
                {
                    "choices": [
                        {"index": 0, "delta": {"content": "Hi <PERSON_1>"}},
                        {"index": 1, "delta": {"content": "Hi <PERSON_1>"}},
                    ]
                }
            ),
            _oai_event("[DONE]"),
        ]
        out = (await _collect(oai.transform_stream(_aiter(chunks), m))).decode()
        # First choice masked → unmasked. Second choice's content passes
        # through unchanged (broken contract for n>1, documented).
        assert "Hi Alice" in out  # first choice
        # The second choice still has its raw `<PERSON_1>` in the original JSON.
        # We don't assert on this — just pin that n>1 is not supported.


@pytest.mark.asyncio
class TestOpenAIPassthrough:
    async def test_done_sentinel_emitted(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store)
        out = await _collect(oai.transform_stream(_aiter([_oai_event("[DONE]")]), m))
        assert b"[DONE]" in out

    async def test_unparseable_event_passes_through(self, make_filter, store):
        m = _make_masker_with_tokens(make_filter, store)
        chunk = b"data: not valid json\n\n"
        out = await _collect(oai.transform_stream(_aiter([chunk]), m))
        assert b"not valid json" in out


# ---------------------------------------------------------------------------
# Phase 4c: decode-failure / passthrough robustness contracts.
#
# Server-level _passthrough (non-POST, empty body, JSON decode failure,
# not-maskable) is out of plan scope (server.py is read-only support).
# These tests cover the adapter-level decode-failure paths.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAnthropicDecodeFailure:
    async def test_invalid_json_in_sse_data_passes_through(self, make_filter, store):
        # If the upstream emits a `data:` line that isn't valid JSON,
        # _transform_event yields data_str unchanged rather than crashing.
        m = _make_masker_with_tokens(make_filter, store, ("PERSON", "Alice"))
        chunk = b"event: content_block_delta\ndata: not valid json\n\n"
        out = await _collect(anth.transform_stream(_aiter([chunk]), m))
        assert b"not valid json" in out
        # Event type preserved too.
        assert b"event: content_block_delta" in out


class TestIsCompleteJson:
    """Pin the _is_complete_json helper that gates tool_call argument
    flushing in the OpenAI adapter. Falsy on invalid JSON, truthy on valid."""

    def test_complete_returns_true(self):
        from anon_proxy.adapters.openai import _is_complete_json

        assert _is_complete_json('{"a": 1}') is True
        assert _is_complete_json("[]") is True
        assert _is_complete_json("null") is True

    def test_incomplete_returns_false(self):
        from anon_proxy.adapters.openai import _is_complete_json

        assert _is_complete_json('{"a":') is False
        assert _is_complete_json("[1,") is False

    def test_garbage_returns_false(self):
        from anon_proxy.adapters.openai import _is_complete_json

        assert _is_complete_json("not json at all") is False
        assert _is_complete_json("") is False
