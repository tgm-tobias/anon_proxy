"""Fail-closed outbound masking.

Every string leaf in an outbound request body is masked unless an explicit
policy entry passes it through. New API fields then fail visibly by being
over-masked instead of invisibly leaking raw text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from anon_proxy.masker import Masker


@dataclass(frozen=True)
class Policy:
    pass_keys: frozenset[str] = frozenset()
    pass_paths: frozenset[tuple[str, ...]] = frozenset()
    pass_block_types: frozenset[str] = frozenset()
    pass_block_subtrees: dict[str, str] = field(default_factory=dict)


def mask_body(body: dict, masker: Masker, policy: Policy) -> dict:
    return {
        key: _walk_path((key,), value, masker, policy) for key, value in body.items()
    }


def _walk_path(
    path: tuple[str, ...], value: Any, masker: Masker, policy: Policy
) -> Any:
    if path in policy.pass_paths:
        return value
    if path == ("messages",) and isinstance(value, list):
        return [masker.mask_obj(m, lambda mm: _walk(mm, masker, policy)) for m in value]
    return _walk_value(path[-1], value, masker, policy)


def _walk_value(key: str, value: Any, masker: Masker, policy: Policy) -> Any:
    if isinstance(value, str):
        return value if key in policy.pass_keys else masker.mask(value)
    if isinstance(value, dict):
        block_type = value.get("type")
        if block_type in policy.pass_block_types:
            return value
        pass_subtree = policy.pass_block_subtrees.get(block_type)
        return {
            k: v if k == pass_subtree else _walk_value(k, v, masker, policy)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_walk(v, masker, policy) for v in value]
    return value


def _walk(value: Any, masker: Masker, policy: Policy) -> Any:
    if isinstance(value, str):
        return masker.mask(value)
    if isinstance(value, dict):
        block_type = value.get("type")
        if block_type in policy.pass_block_types:
            return value
        pass_subtree = policy.pass_block_subtrees.get(block_type)
        return {
            k: v if k == pass_subtree else _walk_value(k, v, masker, policy)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_walk(v, masker, policy) for v in value]
    return value
