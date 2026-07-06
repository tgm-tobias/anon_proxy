from __future__ import annotations

import re

import pytest

from anon_proxy.default_patterns import DEFAULT_PATTERNS


POSITIVE = [
    ("EMAIL", "alice.smith+tag@sub.example.co.uk"),
    ("PHONE", "+1 (415) 555-2671"),
    ("PHONE", "415-555-2671"),
    ("PHONE", "442 222 47571"),
    ("SSN", "078-05-1120"),
    ("IPV4", "10.42.4.43"),
    ("CREDIT_CARD", "4111 1111 1111 1111"),
    ("AWS_ACCESS_KEY", "AKIAIOSFODNN7EXAMPLE"),
    ("GITHUB_TOKEN", "ghp_16CharactersOfEntropy0123456789abcd"),
    (
        "JWT",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0."
        "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk",
    ),
    ("PRIVATE_KEY", "-----BEGIN RSA PRIVATE KEY-----"),
]

NEGATIVE = [
    ("PHONE", "torch==2.11.0"),
    ("PHONE", "1234"),
    ("SSN", "127-0-1"),
    ("IPV4", "999.999.999.999"),
    ("CREDIT_CARD", "1234 5678"),
    ("EMAIL", "not-an-email@"),
    ("GITHUB_TOKEN", "ghp_short"),
]


@pytest.mark.parametrize(("label", "text"), POSITIVE)
def test_pattern_matches(label: str, text: str) -> None:
    assert re.search(DEFAULT_PATTERNS[label], text), (label, text)


@pytest.mark.parametrize(("label", "text"), NEGATIVE)
def test_pattern_rejects(label: str, text: str) -> None:
    assert not re.search(DEFAULT_PATTERNS[label], text), (label, text)
