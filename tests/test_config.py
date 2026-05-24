"""Tests for `load_config` — focused on the `system_inject` knob.

The other Config fields (patterns/merge_gap/ignore_labels) are exercised
indirectly through the masker/detector tests. This file pins the parsing
contract for `system_inject` specifically since it's the newest addition and
the CLI flag layers on top of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anon_proxy.config import Config, load_config


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


class TestSystemInject:
    def test_default_is_true_when_field_absent(self, tmp_path):
        cfg = load_config(_write(tmp_path, {}))
        assert cfg.system_inject is True

    def test_dataclass_default_is_true(self):
        # The bare dataclass default has to match load_config's default — the
        # CLI path constructs `Config()` directly when no --config is given.
        assert Config().system_inject is True

    def test_explicit_false_disables(self, tmp_path):
        cfg = load_config(_write(tmp_path, {"system_inject": False}))
        assert cfg.system_inject is False

    def test_explicit_true_enables(self, tmp_path):
        cfg = load_config(_write(tmp_path, {"system_inject": True}))
        assert cfg.system_inject is True

    def test_string_rejected(self, tmp_path):
        # "false" as a string is a common typo; we reject so it doesn't
        # silently get interpreted as truthy.
        with pytest.raises(ValueError, match="system_inject"):
            load_config(_write(tmp_path, {"system_inject": "false"}))

    def test_int_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="system_inject"):
            load_config(_write(tmp_path, {"system_inject": 0}))

    def test_unknown_key_still_rejected(self, tmp_path):
        # Regression guard: adding `system_inject` to the allowed-keys set
        # must not have weakened the unknown-key check.
        with pytest.raises(ValueError, match="unknown top-level keys"):
            load_config(_write(tmp_path, {"bogus": True}))
