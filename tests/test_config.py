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
from anon_proxy.upstream import UpstreamConfig


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


class TestUpstreams:
    def test_default_is_empty(self, tmp_path):
        cfg = load_config(_write(tmp_path, {}))
        assert cfg.upstreams == {}

    def test_minimal_entry_applies_defaults(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                {"upstreams": {"deepseek": {"base_url": "https://api.deepseek.com"}}},
            )
        )
        assert cfg.upstreams == {
            "deepseek": UpstreamConfig(
                name="deepseek",
                base_url="https://api.deepseek.com",
                path_prefix="",
                adapter="anthropic",
                sse=True,
            )
        }

    def test_full_entry_round_trip(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                {
                    "upstreams": {
                        "zai": {
                            "base_url": "https://api.z.ai/",
                            "adapter": "anthropic",
                            "path_prefix": "api/anthropic",
                            "sse": False,
                        }
                    }
                },
            )
        )
        # Trailing slash on base_url should be stripped.
        assert cfg.upstreams["zai"] == UpstreamConfig(
            name="zai",
            base_url="https://api.z.ai",
            path_prefix="api/anthropic",
            adapter="anthropic",
            sse=False,
        )

    def test_missing_base_url_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="base_url"):
            load_config(_write(tmp_path, {"upstreams": {"x": {}}}))

    def test_empty_base_url_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="base_url"):
            load_config(_write(tmp_path, {"upstreams": {"x": {"base_url": ""}}}))

    def test_bad_adapter_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="adapter"):
            load_config(
                _write(
                    tmp_path,
                    {"upstreams": {"x": {"base_url": "https://a", "adapter": "grpc"}}},
                )
            )

    def test_unknown_subkey_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown keys"):
            load_config(
                _write(
                    tmp_path,
                    {"upstreams": {"x": {"base_url": "https://a", "bogus": 1}}},
                )
            )

    def test_non_object_spec_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_config(_write(tmp_path, {"upstreams": {"x": "https://a"}}))

    def test_non_bool_sse_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="sse"):
            load_config(
                _write(
                    tmp_path,
                    {"upstreams": {"x": {"base_url": "https://a", "sse": "yes"}}},
                )
            )

    def test_top_level_must_be_object(self, tmp_path):
        with pytest.raises(ValueError, match="upstreams"):
            load_config(_write(tmp_path, {"upstreams": []}))
