"""Tests for server-level persistence wiring.

Covered:
- ``_write_store_json`` — raw file-writing helper (sync, runs in thread pool).
- ``_maybe_save_store`` — the async gate that decides whether to write and
  offloads I/O to a thread.
"""

from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.testclient import TestClient

from anon_proxy.adapters import anthropic as anthropic_adapter
from anon_proxy.mapping import PIIStore
from anon_proxy.masker import Masker
from anon_proxy.registry import MaskerRegistry
from anon_proxy.server import (
    _extract_usage,
    build_app,
    _maybe_save_store,
    _should_mask_request,
    _upstream_request,
    _write_store_json,
)
from anon_proxy.upstream import UpstreamConfig
from tests.conftest import span


# ---------------------------------------------------------------------------
# _write_store_json (the sync I/O helper)
# ---------------------------------------------------------------------------


class TestWriteStoreJson:
    def test_writes_valid_store_file(self, tmp_path):
        path = tmp_path / "store.json"
        data = {"reverse": {"<PERSON_1>": "Alice"}, "counters": {"PERSON": 2}}
        _write_store_json(str(path), data)
        assert path.exists()
        loaded = PIIStore.load(str(path))
        assert loaded.original("<PERSON_1>") == "Alice"

    def test_tmp_file_cleaned_up(self, tmp_path):
        path = tmp_path / "store.json"
        _write_store_json(str(path), {"reverse": {}, "counters": {}})
        assert not (tmp_path / "store.json.tmp").exists()

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "store.json"
        _write_store_json(
            str(path), {"reverse": {"<P_1>": "first"}, "counters": {"P": 2}}
        )
        assert PIIStore.load(str(path)).original("<P_1>") == "first"
        _write_store_json(
            str(path), {"reverse": {"<P_1>": "second"}, "counters": {"P": 2}}
        )
        assert PIIStore.load(str(path)).original("<P_1>") == "second"

    def test_non_existent_directory_raises(self, tmp_path):
        path = tmp_path / "missing" / "store.json"
        with pytest.raises(OSError):
            _write_store_json(str(path), {"reverse": {}, "counters": {}})


# ---------------------------------------------------------------------------
# _maybe_save_store (the async gate)
# ---------------------------------------------------------------------------


class TestMaybeSaveStore:
    async def test_saves_when_store_grew(self, tmp_path):
        store_path = str(tmp_path / "store.json")
        store = PIIStore()
        store.get_or_create("PERSON", "Alice")

        await _maybe_save_store(
            SimpleNamespace(store=store),
            store_path,
            store_before=0,
        )
        assert os.path.exists(store_path)
        assert PIIStore.load(store_path).original("<PERSON_1>") == "Alice"

    async def test_does_not_save_when_store_unchanged(self, tmp_path):
        store_path = str(tmp_path / "store.json")
        store = PIIStore()
        store.get_or_create("PERSON", "Alice")

        # store_before=1 means "the store already had 1 entry before the request"
        await _maybe_save_store(
            SimpleNamespace(store=store),
            store_path,
            store_before=1,
        )
        assert not os.path.exists(store_path)

    async def test_no_store_path_skips_save(self, tmp_path):
        store = PIIStore()
        store.get_or_create("PERSON", "Alice")

        await _maybe_save_store(
            SimpleNamespace(store=store),
            None,
            store_before=0,
        )
        # Should not raise and should not create anything

    async def test_multiple_growths_all_saved(self, tmp_path):
        store_path = str(tmp_path / "store.json")
        store = PIIStore()

        # First request — one new entry
        store.get_or_create("PERSON", "Alice")
        await _maybe_save_store(
            SimpleNamespace(store=store),
            store_path,
            store_before=0,
        )
        assert PIIStore.load(store_path).original("<PERSON_1>") == "Alice"

        # Second request — another entry
        store.get_or_create("EMAIL", "a@b.com")
        await _maybe_save_store(
            SimpleNamespace(store=store),
            store_path,
            store_before=1,
        )
        loaded = PIIStore.load(store_path)
        assert loaded.original("<PERSON_1>") == "Alice"
        assert loaded.original("<EMAIL_1>") == "a@b.com"

    async def test_io_error_caught_and_logged(self, tmp_path):
        """OSError from the write is swallowed, never propagates."""
        store_path = str(tmp_path / "no-such-dir" / "store.json")
        store = PIIStore()
        store.get_or_create("PERSON", "Alice")

        await _maybe_save_store(
            SimpleNamespace(store=store),
            store_path,
            store_before=0,
        )
        # Should not raise


# ====================================================================# _should_mask_request
# =============================================================# ---------------------------------------------------------------------------
# build_app multi-user masking
# ---------------------------------------------------------------------------


def _anthropic_echo_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={"content": [{"type": "text", "text": "Hello <PERSON_1>"}]},
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://upstream.example/v1/messages"),
    )


class TestMultiUserProxy:
    def test_multi_user_requires_credential(self, make_filter):
        registry = MaskerRegistry(
            lambda store: Masker(filter=make_filter(), store=store),
            store_dir=None,
        )
        app = build_app(registry=registry)

        with TestClient(app) as client:
            resp = client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert resp.status_code == 401
        assert "x-api-key" in resp.json()["error"]

    def test_multi_user_clients_are_isolated(self, make_filter, fake_pipeline):
        fake_pipeline.set("I am Alice", [span("private_person", 5, 10, word="Alice")])
        registry = MaskerRegistry(
            lambda store: Masker(filter=make_filter(), store=store),
            store_dir=None,
        )
        app = build_app(
            registry=registry,
            extra_upstreams={
                "stub": UpstreamConfig(
                    name="stub",
                    base_url="https://upstream.example",
                    path_prefix="",
                    adapter="anthropic",
                    sse=True,
                )
            },
            system_inject=False,
        )

        with patch(
            "anon_proxy.server._upstream_request",
            AsyncMock(
                side_effect=[_anthropic_echo_response(), _anthropic_echo_response()]
            ),
        ):
            with TestClient(app) as client:
                resp_a = client.post(
                    "/stub/v1/messages",
                    headers={"x-api-key": "A"},
                    json={
                        "model": "claude",
                        "messages": [{"role": "user", "content": "I am Alice"}],
                    },
                )
                resp_b = client.post(
                    "/stub/v1/messages",
                    headers={"x-api-key": "B"},
                    json={
                        "model": "claude",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )

        assert resp_a.status_code == 200
        assert resp_b.status_code == 200
        assert resp_a.json()["content"][0]["text"] == "Hello Alice"
        assert resp_b.json()["content"][0]["text"] == "Hello <PERSON_1>"


# ====================================================================# _should_mask_request
# ====================================================================


class TestShouldMaskRequest:
    """Pure function: decides whether a request needs PII masking."""

    def test_count_tokens_with_messages_is_masked(self):
        # count_tokens carries the full conversation history; the bytes leave
        # the box, so it must be masked like /v1/messages itself.
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert _should_mask_request("/v1/messages/count_tokens", body) is True

    def test_count_tokens_with_provider_prefix_is_masked(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert _should_mask_request("/anthropic/v1/messages/count_tokens", body) is True

    def test_count_tokens_without_pii_fields_not_masked(self):
        assert _should_mask_request("/v1/messages/count_tokens", {}) is False

    def test_messages_endpoint(self):
        assert _should_mask_request("/v1/messages", {"model": "sonnet"}) is True

    def test_chat_completions_endpoint(self):
        assert _should_mask_request("/chat/completions", {"model": "gpt-4"}) is True

    def test_body_with_messages_field(self):
        assert (
            _should_mask_request(
                "/v1/messages?beta=true", {"messages": [{"role": "user"}]}
            )
            is True
        )

    def test_body_with_prompt_field(self):
        assert _should_mask_request("/v1/completions", {"prompt": "Hello"}) is True

    def test_body_without_pii_fields(self):
        assert _should_mask_request("/v1/models", {"model": "sonnet"}) is False

    def test_empty_body(self):
        assert _should_mask_request("/v1/messages", {}) is True

    def test_count_tokens_substring_safety(self):
        """'count_tokens' as a path segment is specific enough to match only
        the metadata endpoint, not normal message paths."""
        assert _should_mask_request("/v1/messages", {"messages": []}) is True


# ====================================================================# _upstream_request
# ====================================================================


def _mock_response(status_code=200, headers=None):
    """Build a minimal object shaped like an httpx.Response for mocking."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.aread = AsyncMock()
    resp.aclose = AsyncMock()
    return resp


@pytest.fixture
def mock_client():
    """An AsyncClient where .send returns 200 by default."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.build_request = MagicMock(return_value=MagicMock())
    ok = _mock_response(200)
    client.send = AsyncMock(return_value=ok)
    return client, ok


class TestUpstreamRequest:
    """Async function: wraps one httpx.AsyncClient.send."""

    @patch("anon_proxy.server.asyncio.sleep", AsyncMock())
    async def test_successful_request(self, mock_client):
        client, ok = mock_client
        resp = await _upstream_request(client, "POST", "https://example.com/api")
        assert resp is ok
        assert resp.status_code == 200
        client.build_request.assert_called_once()
        client.send.assert_awaited_once()

    async def test_429_passes_through_with_retry_after(self, mock_client):
        client, _ok = mock_client
        err = _mock_response(429)
        err.headers = {"retry-after": "7"}
        client.send.return_value = err

        resp = await _upstream_request(client, "POST", "https://example.com/api")
        assert resp.status_code == 429
        assert resp.headers["retry-after"] == "7"
        client.build_request.assert_called_once()
        client.send.assert_awaited_once()
        err.aread.assert_not_awaited()
        err.aclose.assert_not_awaited()

    @patch("anon_proxy.server.asyncio.sleep", AsyncMock())
    async def test_passthrough_args_to_build_request(self, mock_client):
        client, ok = mock_client
        await _upstream_request(
            client,
            "GET",
            "https://example.com/resource",
            content=b'{"key": "val"}',
            headers={"Authorization": "Bearer xyz"},
            params={"page": "1"},
            stream=False,
        )
        client.build_request.assert_called_once_with(
            "GET",
            "https://example.com/resource",
            content=b'{"key": "val"}',
            headers={"Authorization": "Bearer xyz"},
            params={"page": "1"},
        )


class TestProxyMaskingConcurrency:
    @pytest.mark.anyio
    async def test_event_loop_not_blocked_during_mask(self, monkeypatch):
        def slow_mask_request(body, masker):
            time.sleep(0.2)
            return body

        async def fake_upstream_request(*_args, **_kwargs):
            response = MagicMock(spec=httpx.Response)
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = {"content": []}
            return response

        monkeypatch.setattr(anthropic_adapter, "mask_request", slow_mask_request)
        monkeypatch.setattr(
            "anon_proxy.server._upstream_request", fake_upstream_request
        )
        app = build_app(
            masker=SimpleNamespace(store=PIIStore(), unmask=lambda text: text),
            system_inject=False,
        )

        async def post_messages(client):
            return await client.post(
                "/anthropic/v1/messages",
                json={
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "hello Alice"}],
                },
            )

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                t0 = time.perf_counter()
                r1, r2 = await asyncio.gather(
                    post_messages(client),
                    post_messages(client),
                )
                elapsed = time.perf_counter() - t0

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert elapsed < 0.35, f"requests serialized on the event loop: {elapsed:.2f}s"


class TestExtractUsage:
    def test_anthropic_usage(self):
        j = {
            "usage": {
                "input_tokens": 900,
                "cache_read_input_tokens": 8000,
                "cache_creation_input_tokens": 120,
                "output_tokens": 50,
            }
        }
        assert _extract_usage(j) == {
            "input": 900,
            "cache_read": 8000,
            "cache_creation": 120,
        }

    def test_openai_usage(self):
        j = {
            "usage": {
                "prompt_tokens": 900,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 700},
            }
        }
        assert _extract_usage(j) == {
            "input": 900,
            "cache_read": 700,
            "cache_creation": 0,
        }

    def test_no_usage_returns_none(self):
        assert _extract_usage({}) is None
