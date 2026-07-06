"""Exact-match detection of values already learned by the placeholder store."""

from __future__ import annotations

import re

from anon_proxy.mapping import PIIStore
from anon_proxy.privacy_filter import PIIEntity

_TOKEN = re.compile(r"^<([A-Z][A-Z0-9_]*)_\d+>$")


class KnownEntityDetector:
    """Find stored original values in clue-less contexts such as code or logs."""

    def __init__(self, store: PIIStore, min_len: int = 6) -> None:
        if min_len < 0:
            raise ValueError("min_len must be >= 0")
        self._store = store
        self._min_len = min_len
        self._built_at = -1
        self._rx: re.Pattern[str] | None = None
        self._label_by_lower: dict[str, str] = {}

    def detect(self, text: str) -> list[PIIEntity]:
        if len(self._store) != self._built_at:
            self._rebuild()
        if self._rx is None or not text:
            return []
        out: list[PIIEntity] = []
        for match in self._rx.finditer(text):
            label = self._label_by_lower.get(match.group(0).casefold())
            if label is None:
                continue
            out.append(
                PIIEntity(
                    label=label,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    score=1.0,
                )
            )
        return out

    def _rebuild(self) -> None:
        pairs: list[tuple[str, str]] = []
        for token, value in self._store.items():
            if len(value) < self._min_len:
                continue
            parsed = _TOKEN.match(token)
            if parsed:
                pairs.append((value, parsed.group(1)))

        self._label_by_lower = {value.casefold(): label for value, label in pairs}
        if pairs:
            alternatives = sorted(
                (re.escape(value) for value, _ in pairs), key=len, reverse=True
            )
            self._rx = re.compile(
                r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE
            )
        else:
            self._rx = None
        self._built_at = len(self._store)
