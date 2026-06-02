"""Tests for server-level persistence wiring.

Covered:
- ``_write_store_json`` — raw file-writing helper (sync, runs in thread pool).
- ``_maybe_save_store`` — the async gate that decides whether to write and
  offloads I/O to a thread.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from anon_proxy.mapping import PIIStore
from anon_proxy.server import _maybe_save_store, _write_store_json


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
