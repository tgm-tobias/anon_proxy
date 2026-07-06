"""High-precision default regex detectors.

These cover clue-less PII the ML model can miss, plus deterministic secret
formats common in coding-agent traffic. False positives corrupt commands, so
the patterns favor structured shapes over broad guesses.
"""

from __future__ import annotations


DEFAULT_PATTERNS: dict[str, str] = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "PHONE": (
        r"(?<![\w.=/])(?:\+\d{1,3}[ .-]?)?"
        r"(?:(?:\(\d{2,4}\)|\d{2,4})[ .-])?"
        r"\d{3}[ .-]\d{3,5}(?:[ .-]\d{1,4})?(?![\w.-])"
    ),
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "IPV4": (
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
    ),
    # Separator-required by design: avoids masking opaque 13-19 digit IDs.
    "CREDIT_CARD": r"\b\d{4}(?:[ -]\d{4}){3}\b",
    "AWS_ACCESS_KEY": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "GITHUB_TOKEN": r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    "JWT": r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
    "PRIVATE_KEY": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "SLACK_TOKEN": r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
}
