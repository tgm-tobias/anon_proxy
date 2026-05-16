"""User-defined regex-based PII detection.

Complements the ML detector for kinds of PII it doesn't reliably catch
(IP addresses, SSNs, credit cards, internal IDs) or that a specific deployment
wants handled deterministically.

Patterns are passed as a flat dict mapping label -> regex string. Loading from
disk is handled by `anon_proxy.config.load_config` (the unified config.json).
"""

from __future__ import annotations

import re

from anon_proxy.privacy_filter import PIIEntity


class RegexDetector:
    """Emits PIIEntity spans for every match of each configured pattern."""

    def __init__(self, patterns: dict[str, str]) -> None:
        compiled: list[tuple[str, re.Pattern[str]]] = []
        errors: list[str] = []
        for label, pattern in patterns.items():
            try:
                compiled.append((label, re.compile(pattern)))
            except re.error as e:
                errors.append(f"  {label!r}: {e}")
        if errors:
            raise ValueError("invalid regex patterns:\n" + "\n".join(errors))
        self._patterns = compiled

    def detect(self, text: str) -> list[PIIEntity]:
        out: list[PIIEntity] = []
        for label, rx in self._patterns:
            for m in rx.finditer(text):
                start, end = m.span()
                if start == end:
                    continue
                out.append(
                    PIIEntity(
                        label=label,
                        text=text[start:end],
                        start=start,
                        end=end,
                        score=1.0,
                    )
                )
        return out

    def __len__(self) -> int:
        return len(self._patterns)
