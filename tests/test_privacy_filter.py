"""Tests for PrivacyFilter.

Specs covered (agreed in Phase 1a):
- Empty / whitespace-only input short-circuits to [] without calling the pipeline.
- detect_batch is removed.
- Spans with zero length after tightening are dropped.
- Label normalization prefers entity_group, falls back to entity.
- Chunking splits at last whitespace before chunk_size, hard-cuts otherwise,
  and spans from each chunk are shifted by their chunk offset.
- Adjacency merging: same label + empty gap merges unconditionally; non-empty
  gap merges iff every gap character is in `merge_gap_allowed[label]`; merged
  score = min(scores); merged span is tightened.
- aggregation_strategy and device remain configurable constructor params.
"""

from __future__ import annotations

import pytest

from anon_proxy import privacy_filter
from anon_proxy.privacy_filter import (
    DEFAULT_MERGE_GAP_ALLOWED,
    PrivacyFilter,
    _gap_mergeable,
    _split_chunks,
    _tighten,
)

from .conftest import span


# ---------------------------------------------------------------------------
# Pure helpers — exercised directly for boundary cases that detect() would
# require contrived chunk-aligned spans to surface.
# ---------------------------------------------------------------------------


class TestSplitChunks:
    def test_empty_text_returns_single_empty_chunk(self):
        assert _split_chunks("", 10) == [(0, "")]

    def test_short_text_returns_single_chunk(self):
        assert _split_chunks("hello", 10) == [(0, "hello")]

    def test_exact_boundary_returns_single_chunk(self):
        assert _split_chunks("abcdefghij", 10) == [(0, "abcdefghij")]

    def test_splits_at_last_whitespace_before_boundary(self):
        # max=10, splits at the last space in [0, 10); space goes with earlier chunk.
        text = "hello world foo"
        chunks = _split_chunks(text, 10)
        assert chunks == [(0, "hello "), (6, "world foo")]

    def test_hard_cut_when_no_whitespace_in_window(self):
        text = "abcdefghijklmno"
        chunks = _split_chunks(text, 5)
        # No spaces anywhere; every chunk is a hard cut of exactly 5 chars
        # until the final tail.
        assert chunks == [(0, "abcde"), (5, "fghij"), (10, "klmno")]

    def test_chunks_cover_full_text(self):
        text = "Alice Smith called from 555-867-5309 about her invoice."
        chunks = _split_chunks(text, 12)
        assert "".join(c for _, c in chunks) == text
        # offsets are monotonic and start of each chunk == sum of prior lengths
        running = 0
        for offset, chunk in chunks:
            assert offset == running
            running += len(chunk)

    def test_last_chunk_may_exceed_max_when_tail_small(self):
        # "abcdefghij k" — chunk_size=10. start=0, start+max=10 >= len(text)? len=12, no.
        # split = rfind(" ", 0, 10) → 10 is exclusive, so we look in [0,10): "abcdefghij" — no space.
        # Hard cut at 10. Next start=10, start+max=20 >= 12 → tail chunk text[10:] = " k" (length 2, fits).
        text = "abcdefghij k"
        chunks = _split_chunks(text, 10)
        assert chunks == [(0, "abcdefghij"), (10, " k")]


class TestTighten:
    def test_no_whitespace_unchanged(self):
        assert _tighten(0, 5, "hello") == (0, 5)

    def test_strips_leading_whitespace(self):
        assert _tighten(0, 6, "  hi  ") == (2, 4)

    def test_strips_trailing_whitespace(self):
        assert _tighten(0, 7, "hello  ") == (0, 5)

    def test_all_whitespace_collapses_to_zero_length(self):
        s, e = _tighten(0, 4, "    ")
        assert s == e  # zero-length

    def test_handles_inner_whitespace_only_at_edges(self):
        # Internal whitespace is preserved; only edges are trimmed.
        assert _tighten(0, 11, "  a b  c   ") == (2, 8)


class TestGapMergeable:
    def test_empty_gap_always_mergeable(self):
        assert _gap_mergeable("ANYTHING", "", None) is True
        assert _gap_mergeable("ANYTHING", "", {}) is True

    def test_missing_label_blocks_non_empty_gap(self):
        assert _gap_mergeable("UNKNOWN", " ", {"OTHER": frozenset(" ")}) is False

    def test_label_with_empty_allowed_blocks_non_empty_gap(self):
        assert _gap_mergeable("LABEL", " ", {"LABEL": frozenset("")}) is False

    def test_all_gap_chars_must_be_allowed(self):
        allowed = {"LABEL": frozenset(" -")}
        assert _gap_mergeable("LABEL", " ", allowed) is True
        assert _gap_mergeable("LABEL", "-", allowed) is True
        assert _gap_mergeable("LABEL", " - ", allowed) is True
        assert _gap_mergeable("LABEL", " x ", allowed) is False  # 'x' not allowed


# ---------------------------------------------------------------------------
# detect() — end-to-end via FakePipeline. These cover the public contract
# and exercise chunking, label normalization, tightening, and merging together.
# ---------------------------------------------------------------------------


class TestDetectEmptyShortCircuit:
    """Empty or whitespace-only inputs must not call the pipeline at all."""

    def test_empty_string_short_circuits(self, make_filter, fake_pipeline):
        f = make_filter()
        assert f.detect("") == []
        assert fake_pipeline.calls == []

    def test_whitespace_only_short_circuits(self, make_filter, fake_pipeline):
        f = make_filter()
        assert f.detect("   \t\n  ") == []
        assert fake_pipeline.calls == []


class TestDetectLabelNormalization:
    def test_prefers_entity_group_when_present(self, make_filter, fake_pipeline):
        text = "Alice"
        fake_pipeline.set(text, [span("PERSON", 0, 5, score=0.9)])
        f = make_filter()
        [e] = f.detect(text)
        assert e.label == "PERSON"

    def test_falls_back_to_entity_when_group_missing(self, make_filter, fake_pipeline):
        text = "Alice"
        fake_pipeline.set(
            text,
            [span("PERSON", 0, 5, score=0.9, use_entity_key=True)],
        )
        f = make_filter()
        [e] = f.detect(text)
        assert e.label == "PERSON"


class TestDetectTighteningInvariant:
    def test_entity_text_matches_original_slice(self, make_filter, fake_pipeline):
        # Pipeline returns a span with leading/trailing whitespace; detect
        # tightens so entity.text == original[start:end].
        text = "Hello  Alice  there"
        fake_pipeline.set(text, [span("PERSON", 5, 14, score=0.9)])  # "  Alice  "
        f = make_filter()
        [e] = f.detect(text)
        assert e.text == text[e.start : e.end]
        assert e.text == "Alice"
        assert e.start == 7
        assert e.end == 12


class TestDetectZeroLengthDropped:
    def test_all_whitespace_span_dropped(self, make_filter, fake_pipeline):
        text = "Hello     world"
        fake_pipeline.set(
            text,
            [
                span("PERSON", 5, 10, score=0.9),  # "     " → zero-length after tighten
                span("PERSON", 10, 15, score=0.9),  # "world" — kept
            ],
        )
        f = make_filter()
        out = f.detect(text)
        assert len(out) == 1
        assert out[0].text == "world"


class TestDetectScoreAndTypes:
    def test_score_is_float_and_indices_int(self, make_filter, fake_pipeline):
        text = "Alice"
        fake_pipeline.set(
            text,
            [
                {
                    "entity_group": "PERSON",
                    "start": "0",
                    "end": "5",
                    "word": "",
                    "score": "0.9",
                }
            ],
        )
        f = make_filter()
        [e] = f.detect(text)
        assert isinstance(e.start, int) and e.start == 0
        assert isinstance(e.end, int) and e.end == 5
        assert isinstance(e.score, float) and e.score == 0.9


class TestDetectIsPlaceholderAgnostic:
    """PrivacyFilter is a pure detector — placeholder-shaped strings like
    `<PERSON_1>` are ordinary text to it. The contract that prevents
    re-masking placeholders lives in Masker._drop_placeholder_overlaps
    (covered in Phase 3c), NOT here.
    """

    def test_placeholder_string_passes_through_unchanged(
        self, make_filter, fake_pipeline
    ):
        # The fake pipeline is asked to flag the placeholder substring; the
        # filter must surface it as-is (no special-casing).
        text = "Hi <PERSON_1> there"
        fake_pipeline.set(text, [span("PERSON", 3, 13, score=0.9)])  # "<PERSON_1>"
        f = make_filter()
        [e] = f.detect(text)
        assert e.text == "<PERSON_1>"
        assert (e.start, e.end) == (3, 13)

    def test_no_detection_inside_placeholder_when_pipeline_silent(
        self, make_filter, fake_pipeline
    ):
        # If the model returns nothing for placeholder-shaped text, the
        # filter returns nothing — it doesn't synthesize spans from `<...>`.
        text = "Hi <PERSON_1> there"
        fake_pipeline.set(text, [])
        f = make_filter()
        assert f.detect(text) == []


# ---------------------------------------------------------------------------
# Chunking via detect() — offset arithmetic and cross-chunk merging.
# ---------------------------------------------------------------------------


class TestDetectChunking:
    def test_offsets_shifted_by_chunk_start(self, make_filter, fake_pipeline):
        # chunk_size=10, text="hello world foo" → chunks = [(0,"hello "), (6,"world foo")]
        text = "hello world foo"
        fake_pipeline.set("hello ", [span("PERSON", 0, 5, score=0.9)])  # "hello"
        fake_pipeline.set(
            "world foo", [span("PERSON", 6, 9, score=0.9)]
        )  # "foo" (idx 6..9 within chunk)
        f = make_filter(chunk_size=10)
        out = f.detect(text)
        # 2 entities at absolute offsets 0..5 and 12..15
        assert [(e.start, e.end, e.text) for e in out] == [
            (0, 5, "hello"),
            (12, 15, "foo"),
        ]

    def test_multichunk_text_is_one_batched_pipeline_call(
        self, make_filter, fake_pipeline
    ):
        text = "hello world foo"
        f = make_filter(chunk_size=10)
        f.detect(text)
        assert len(fake_pipeline.calls) == 1
        assert isinstance(fake_pipeline.calls[0], list)
        assert "".join(fake_pipeline.calls[0]) == text

    def test_batched_multichunk_offsets_shift_by_chunk_start(
        self, make_filter, fake_pipeline
    ):
        text = "aaaa bbbb cccc dddd"
        f = make_filter(chunk_size=10)
        chunks = _split_chunks(text, 10)
        assert len(chunks) > 1
        second_offset, second_chunk = chunks[1]
        fake_pipeline.set(second_chunk, [span("PERSON", 0, 4, score=0.9)])

        [entity] = f.detect(text)

        assert entity.text == second_chunk[:4]
        assert entity.start == second_offset
        assert entity.end == second_offset + 4

    def test_cross_chunk_adjacency_merge(self, make_filter, fake_pipeline):
        # chunk_size=7, text="Alice Smith":
        #   rfind(" ", 0, 7) = 5 → split at 6.
        #   chunks = [(0, "Alice "), (6, "Smith")]
        # Each chunk yields one PERSON span; the gap between them in the
        # original is " " (allowed for PERSON), so they merge.
        text = "Alice Smith"
        fake_pipeline.set("Alice ", [span("PERSON", 0, 5, score=0.8)])
        fake_pipeline.set("Smith", [span("PERSON", 0, 5, score=0.7)])
        f = make_filter(chunk_size=7)
        out = f.detect(text)
        assert len(out) == 1
        merged = out[0]
        assert merged.label == "PERSON"
        assert merged.text == "Alice Smith"
        assert (merged.start, merged.end) == (0, 11)
        assert merged.score == pytest.approx(0.7)  # min(scores)


# ---------------------------------------------------------------------------
# Adjacency merging behavior driven through detect().
# ---------------------------------------------------------------------------


class TestDetectAdjacencyMerging:
    def test_empty_gap_merges(self, make_filter, fake_pipeline):
        text = "Alicesmith"
        fake_pipeline.set(
            text,
            [
                span("PERSON", 0, 5, score=0.9),
                span("PERSON", 5, 10, score=0.8),
            ],
        )
        f = make_filter()
        out = f.detect(text)
        assert len(out) == 1
        assert out[0].text == "Alicesmith"
        assert out[0].score == pytest.approx(0.8)

    def test_allowed_gap_chars_merge(self, make_filter, fake_pipeline):
        text = "Alice Smith"
        fake_pipeline.set(
            text,
            [
                span("PERSON", 0, 5, score=0.9),
                span("PERSON", 6, 11, score=0.8),
            ],
        )
        f = make_filter()
        [e] = f.detect(text)
        assert e.text == "Alice Smith"

    def test_disallowed_gap_chars_block_merge(self, make_filter, fake_pipeline):
        # ORGANIZATION's default allowed set doesn't include '/'.
        text = "ACME/Globex"
        fake_pipeline.set(
            text,
            [
                span("ORGANIZATION", 0, 4, score=0.9),
                span("ORGANIZATION", 5, 11, score=0.8),
            ],
        )
        f = make_filter()
        out = f.detect(text)
        assert len(out) == 2

    def test_different_labels_never_merge(self, make_filter, fake_pipeline):
        text = "Alice Smith"
        fake_pipeline.set(
            text,
            [
                span("PERSON", 0, 5, score=0.9),
                span("ORGANIZATION", 6, 11, score=0.8),
            ],
        )
        f = make_filter()
        out = f.detect(text)
        assert {e.label for e in out} == {"PERSON", "ORGANIZATION"}

    def test_user_gap_override_extends_default(self, make_filter, fake_pipeline):
        # Add '/' to ORGANIZATION's allowed set via merge_gap_allowed override.
        text = "ACME/Globex"
        fake_pipeline.set(
            text,
            [
                span("ORGANIZATION", 0, 4, score=0.9),
                span("ORGANIZATION", 5, 11, score=0.8),
            ],
        )
        merged_set = DEFAULT_MERGE_GAP_ALLOWED["ORGANIZATION"] + "/"
        f = make_filter(merge_gap_allowed={"ORGANIZATION": merged_set})
        [e] = f.detect(text)
        assert e.text == "ACME/Globex"

    def test_empty_allowed_disables_gap_merging_for_label(
        self, make_filter, fake_pipeline
    ):
        # Setting PERSON's allowed set to "" should block "Alice Smith" merging.
        text = "Alice Smith"
        fake_pipeline.set(
            text,
            [
                span("PERSON", 0, 5, score=0.9),
                span("PERSON", 6, 11, score=0.8),
            ],
        )
        f = make_filter(merge_gap_allowed={"PERSON": ""})
        out = f.detect(text)
        assert len(out) == 2

    def test_merge_adjacent_false_disables_all_merging(
        self, make_filter, fake_pipeline
    ):
        text = "Alice Smith"
        fake_pipeline.set(
            text,
            [
                span("PERSON", 0, 5, score=0.9),
                span("PERSON", 6, 11, score=0.8),
            ],
        )
        f = make_filter(merge_adjacent=False)
        out = f.detect(text)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Constructor surface.
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_aggregation_strategy_passed_to_pipeline(self, monkeypatch, fake_pipeline):
        captured: dict = {}

        def _stub(**kwargs):
            captured.update(kwargs)
            return fake_pipeline

        monkeypatch.setattr(privacy_filter, "pipeline", _stub)
        PrivacyFilter(aggregation_strategy="max")
        assert captured["aggregation_strategy"] == "max"

    def test_device_passed_to_pipeline(self, monkeypatch, fake_pipeline):
        captured: dict = {}

        def _stub(**kwargs):
            captured.update(kwargs)
            return fake_pipeline

        monkeypatch.setattr(privacy_filter, "pipeline", _stub)
        PrivacyFilter(device="cuda:0")
        assert captured["device"] == "cuda:0"


# ---------------------------------------------------------------------------
# detect_batch — must be removed.
# ---------------------------------------------------------------------------


class TestDetectBatchRemoved:
    def test_detect_batch_attribute_absent(self, make_filter):
        f = make_filter()
        assert not hasattr(f, "detect_batch")
