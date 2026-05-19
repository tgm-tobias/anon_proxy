"""Shared test fixtures.

The HF pipeline is mocked everywhere — tests must never load the real model.
`make_filter` monkeypatches `anon_proxy.privacy_filter.pipeline` to a callable
stub that returns the same FakePipeline instance the test holds, so tests
inject canned spans via `fake_pipeline.set(text, [...])`.
"""

from __future__ import annotations

from typing import Iterable

import pytest

from anon_proxy import privacy_filter
from anon_proxy.mapping import PIIStore
from anon_proxy.privacy_filter import PrivacyFilter


class FakePipeline:
    """Stand-in for transformers.pipeline (token-classification, aggregated).

    Callable shapes mirror the real pipeline:
        fake(text: str)            -> list[dict]
        fake(texts: Iterable[str]) -> list[list[dict]]

    Tests register canned spans per input text. Unregistered inputs yield [].
    """

    def __init__(self) -> None:
        self._responses: dict[str, list[dict]] = {}
        self.calls: list[str | list[str]] = []

    def set(self, text: str, spans: list[dict]) -> None:
        self._responses[text] = spans

    def __call__(self, inputs):
        if isinstance(inputs, str):
            self.calls.append(inputs)
            return list(self._responses.get(inputs, []))
        as_list = list(inputs)
        self.calls.append(as_list)
        return [list(self._responses.get(t, [])) for t in as_list]


def span(
    label: str,
    start: int,
    end: int,
    *,
    word: str = "",
    score: float = 0.99,
    use_entity_key: bool = False,
) -> dict:
    """Build a pipeline-shaped span dict.

    `use_entity_key=True` emits `"entity"` instead of `"entity_group"` so tests
    can exercise the label-key fallback in `_to_entity`.
    """
    key = "entity" if use_entity_key else "entity_group"
    return {key: label, "start": start, "end": end, "word": word, "score": score}


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def make_filter(monkeypatch, fake_pipeline):
    """Construct a PrivacyFilter wired to the test's FakePipeline.

    Usage:
        f = make_filter()                         # defaults
        f = make_filter(chunk_size=20, merge_adjacent=False)
    """

    def _stub_pipeline(**_kw):
        return fake_pipeline

    monkeypatch.setattr(privacy_filter, "pipeline", _stub_pipeline)

    def _make(**kwargs) -> PrivacyFilter:
        return PrivacyFilter(**kwargs)

    return _make


@pytest.fixture
def store() -> PIIStore:
    return PIIStore()
