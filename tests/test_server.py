"""Tests for server-level persistence wiring.

Covered:
- ``_write_store_json`` — raw file-writing helper (sync, runs in thread pool).
- ``_maybe_save_store`` — the async gate that decides whether to write and
  offloads I/O to a thread.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from anon_proxy.mapping import PIIStore
from anon_proxy.server import (
    _maybe_save_store,
    _parse_retry_after,
    _should_mask_request,
    _upstream_request,
    _write_store_json,
)


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

# Helper to build the lightweight state object ``_maybe_save_store`` expects.
_state = SimpleNamespace  # alias for compact tests


class TestMaybeSaveStore:
    async def test_saves_when_store_grew(self, tmp_path):
        store_path = str(tmp_path / "store.json")
        store = PIIStore()
        store.get_or_create("PERSON", "Alice")

        await _maybe_save_store(
            _state(store_path=store_path, masker=_state(store=store)),
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
            _state(store_path=store_path, masker=_state(store=store)),
            store_before=1,
        )
        assert not os.path.exists(store_path)

    async def test_no_store_path_skips_save(self, tmp_path):
        store = PIIStore()
        store.get_or_create("PERSON", "Alice")

        await _maybe_save_store(
            _state(store_path=None, masker=_state(store=store)),
            store_before=0,
        )
        # Should not raise and should not create anything

    async def test_multiple_growths_all_saved(self, tmp_path):
        store_path = str(tmp_path / "store.json")
        store = PIIStore()

        # First request — one new entry
        store.get_or_create("PERSON", "Alice")
        await _maybe_save_store(
            _state(store_path=store_path, masker=_state(store=store)),
            store_before=0,
        )
        assert PIIStore.load(store_path).original("<PERSON_1>") == "Alice"

        # Second request — another entry
        store.get_or_create("EMAIL", "a@b.com")
        await _maybe_save_store(
            _state(store_path=store_path, masker=_state(store=store)),
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
            _state(store_path=store_path, masker=_state(store=store)),
            store_before=0,
        )
        # Should not raise


# ===========================================================================
# _parse_retry_after
# ===========================================================================


class TestParseRetryAfter:
    """Pure function: parses Retry-After header into seconds."""

    def test_retry_after_seconds(self):
        headers = {"retry-after": "5"}
        assert _parse_retry_after(headers) == 5.0

    def test_retry_after_float(self):
        headers = {"retry-after": "2.5"}
        assert _parse_retry_after(headers) == 2.5

    def test_no_retry_after_header(self):
        assert _parse_retry_after({}) is None

    def test_retry_after_invalid_returns_none(self):
        headers = {"retry-after": "foobar"}
        assert _parse_retry_after(headers) is None

    def test_retry_after_case_insensitive(self):
        headers = {"Retry-After": "3"}
        assert _parse_retry_after(headers) == 3.0

    def test_retry_after_both_forms(self):
        """When both variants are present, lowercase wins (dict order)."""
        headers = {"retry-after": "1", "Retry-After": "10"}
        assert _parse_retry_after(headers) == 1.0


# ===========================================================================
# _should_mask_request
# ===========================================================================


class TestShouldMaskRequest:
    """Pure function: decides whether a request needs PII masking."""

    def test_count_tokens_path_returns_false(self):
        assert (
            _should_mask_request("v1/messages/count_tokens", {"messages": []}) is False
        )

    def test_count_tokens_with_provider_prefix(self):
        assert (
            _should_mask_request(
                "/anthropic/v1/messages/count_tokens", {"messages": []}
            )
            is False
        )

    def test_count_tokens_with_messages_body_skipped(self):
        """Path check wins — even a body with PII fields is skipped."""
        assert (
            _should_mask_request("/v1/messages/count_tokens", {"messages": []}) is False
        )

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


# ===========================================================================
# _upstream_request
# ===========================================================================


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
    """Async function: wraps httpx.AsyncClient.send with 429 retry."""

    @patch("anon_proxy.server.asyncio.sleep", AsyncMock())
    async def test_successful_request(self, mock_client):
        client, ok = mock_client
        resp = await _upstream_request(client, "POST", "https://example.com/api")
        assert resp is ok
        assert resp.status_code == 200
        client.build_request.assert_called_once()
        client.send.assert_awaited_once()

    @patch("anon_proxy.server.asyncio.sleep", AsyncMock())
    async def test_single_429_then_success(self, mock_client):
        client, ok = mock_client
        err = _mock_response(429)
        client.send.side_effect = [err, ok]

        resp = await _upstream_request(client, "POST", "https://example.com/api")
        assert resp is ok
        assert resp.status_code == 200
        # Two attempts: 429, then 200
        assert client.send.await_count == 2
        err.aclose.assert_awaited_once()

    @patch("anon_proxy.server.asyncio.sleep", AsyncMock())
    async def test_exhausts_retries_returns_last_429(self, mock_client):
        client, ok = mock_client
        errs = [_mock_response(429) for _ in range(4)]
        client.send.side_effect = list(errs)

        resp = await _upstream_request(client, "POST", "https://example.com/api")
        assert resp.status_code == 429
        assert client.send.await_count == 4  # initial + 3 retries
        # The first 3 responses are drained and closed during retry;
        # the 4th is returned to the caller (caller owns cleanup).
        for e in errs[:-1]:
            e.aclose.assert_awaited_once()
        errs[-1].aclose.assert_not_awaited()

    async def test_respects_retry_after_header(self, mock_client):
        client, ok = mock_client
        err = _mock_response(429, {"retry-after": "2"})
        client.send.side_effect = [err, ok]

        with patch("anon_proxy.server.asyncio.sleep", AsyncMock()) as mock_sleep:
            resp = await _upstream_request(client, "POST", "https://example.com/api")
        assert resp.status_code == 200
        mock_sleep.assert_awaited_once_with(2.0)

    async def test_exponential_backoff_fallback(self, mock_client):
        """When Retry-After is absent, use exponential backoff with jitter."""
        client, ok = mock_client
        err = _mock_response(429)  # no retry-after
        client.send.side_effect = [err, ok]

        with (
            patch("anon_proxy.server.random.random", return_value=0.5),
            patch("anon_proxy.server.asyncio.sleep", AsyncMock()) as mock_sleep,
        ):
            resp = await _upstream_request(client, "POST", "https://example.com/api")
        assert resp.status_code == 200
        # attempt 0: 2^0 * (0.5 + 0.5 * 0.5) = 1.0 * 0.75 = 0.75
        mock_sleep.assert_awaited_once_with(0.75)

    @patch("anon_proxy.server.asyncio.sleep", AsyncMock())
    async def test_streaming_drains_before_retry(self, mock_client):
        """Streaming 429 responses must be .aread() before .aclose()."""
        client, ok = mock_client
        err = _mock_response(429)
        client.send.side_effect = [err, ok]

        await _upstream_request(client, "POST", "https://example.com/api", stream=True)
        err.aread.assert_awaited_once()
        err.aclose.assert_awaited_once()

    @patch("anon_proxy.server.asyncio.sleep", AsyncMock())
    async def test_non_streaming_skips_aread(self, mock_client):
        """Non-streaming 429 responses don't need .aread()."""
        client, ok = mock_client
        err = _mock_response(429)
        client.send.side_effect = [err, ok]

        await _upstream_request(client, "POST", "https://example.com/api", stream=False)
        err.aread.assert_not_awaited()
        err.aclose.assert_awaited_once()

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

    async def test_max_retries_parameter(self, mock_client):
        """Custom max_retries limits the number of retries."""
        client, ok = mock_client
        err = _mock_response(429)
        client.send.side_effect = [err, err, ok]

        resp = await _upstream_request(
            client, "POST", "https://example.com/api", max_retries=1
        )
        assert resp.status_code == 429  # gave up after 1 retry
        assert client.send.await_count == 2  # initial + 1 retry
