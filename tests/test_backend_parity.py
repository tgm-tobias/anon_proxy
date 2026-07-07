"""Parity gate: the onnx backend vs the torch reference, on the real model.

The privacy invariant is coverage, not exact-string equality: the onnx backend
may split or merge spans differently from torch (e.g. torch yields 'Alice' +
'Smith' while onnx yields one 'Alice Smith'), but every character torch masks
MUST also be masked by onnx. Over-masking is allowed (and logged); missing any
character torch masks fails the gate — that would be a fast privacy leak.

Opt-in (downloads gigabytes incl. the ~0.77 GB q4f16 export, minutes of
runtime):

    ANON_PROXY_LIVE_TESTS=1 uv run --extra onnx pytest tests/test_backend_parity.py -v
"""

from __future__ import annotations

import os

import pytest

from anon_proxy.privacy_filter import PrivacyFilter

pytestmark = pytest.mark.skipif(
    os.environ.get("ANON_PROXY_LIVE_TESTS") != "1",
    reason="live-model test; set ANON_PROXY_LIVE_TESTS=1",
)

# One entry per failure mode: every label kind, multi-word merges, a
# chunk-boundary straddle, code context, unicode/CJK, a false-positive check.
GOLDEN = [
    "My name is Alice Smith, reach me at alice.smith@example.com.",
    "Call Jean-Luc O'Neil at 555-867-5309 before Friday.",
    "Ship it to 123 Main St., Apt #4, Springfield IL 62704.",
    "Meeting moved to Jan 3, 2026 with Dr. Maria Gonzalez-Ruiz.",
    "Acme Corp & Sons acquired Globex; contact legal@acme-corp.example.",
    "def notify(user):\n    send(to='bob.jones@example.org', by=user.phone)\n",
    "账户持有人：王小明，电话 +86 138 0013 8000。",
    "The quarterly report shows no anomalies across all regions.",  # no PII
    "Visit https://intranet.example.com/profile/alice-smith for details.",
    # ~7 KB text forcing multiple chunks with PII straddling a chunk boundary.
    ("x " * 1000) + "Alice Smith lives at 9 Elm Road. " + ("y " * 200),
]


def _covered(pf: PrivacyFilter, text: str) -> set[int]:
    """Character indices of `text` masked by pf.detect()."""
    covered: set[int] = set()
    for e in pf.detect(text):
        covered.update(range(e.start, e.end))
    return covered


@pytest.fixture(scope="module")
def torch_pf() -> PrivacyFilter:
    return PrivacyFilter(backend="torch")


@pytest.fixture(scope="module")
def onnx_pf() -> PrivacyFilter:
    return PrivacyFilter(backend="onnx")


def test_onnx_covers_every_char_torch_masks(torch_pf, onnx_pf):
    missed: list[tuple[str, str]] = []
    over: list[tuple[str, str]] = []
    for text in GOLDEN:
        ref = _covered(torch_pf, text)
        got = _covered(onnx_pf, text)
        for i in sorted(ref - got):
            missed.append((text[:40], text[i]))
        extra = got - ref
        if extra:
            over.append((text[:40], "".join(text[i] for i in sorted(extra))))
    if over:
        # Over-masking is safe — record it so a reviewer can eyeball noise.
        print(f"[onnx] over-detections vs torch (safe): {over}")
    assert not missed, f"[onnx] chars torch masks but onnx leaks: {missed}"


def test_onnx_finds_the_obvious_pii(onnx_pf):
    """Guard against a vacuous parity pass (both backends silent)."""
    text = "My name is Alice Smith, reach me at alice.smith@example.com."
    covered = _covered(onnx_pf, text)
    # The email and the name must both be masked.
    email = "alice.smith@example.com"
    email_start = text.index(email)
    assert all(i in covered for i in range(email_start, email_start + len(email)))
    assert text.index("Alice") in covered
