"""Shared test fixtures.

The HF pipeline is mocked everywhere — tests must never load the real model.
`make_filter` monkeypatches `anon_proxy.privacy_filter.pipeline` to a callable
stub that returns the same FakePipeline instance the test holds, so tests
inject canned spans via `fake_pipeline.set(text, [...])`.
"""

from __future__ import annotations


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

    def __call__(self, inputs, **_kwargs):
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


@pytest.fixture
def make_masker(make_filter, store):
    """Construct a Masker wired to the test's FakePipeline.

    Defaults:
      - skip_patterns=[] so the production skip-pattern list doesn't bypass
        tests that put system-reminder-shaped text into a fixture.
      - shares the test's `store` fixture so assertions can inspect it.

    Usage:
        m = make_masker()
        m = make_masker(extra_detectors=[RegexDetector({"X": r"\\d+"})])
        m = make_masker(filter_kwargs={"chunk_size": 50})
    """
    from anon_proxy.masker import Masker

    def _make(
        *,
        filter_kwargs=None,
        extra_detectors=None,
        ignore_labels=None,
        skip_patterns=None,
        cache_size: int = 4096,
    ):
        f = make_filter(**(filter_kwargs or {}))
        return Masker(
            filter=f,
            store=store,
            extra_detectors=extra_detectors or [],
            skip_patterns=skip_patterns if skip_patterns is not None else [],
            ignore_labels=ignore_labels,
            cache_size=cache_size,
        )

    return _make
