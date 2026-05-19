"""Tests for Masker — Phase 3.

Sub-phases will accumulate here:
- 3a (this section): _resolve_overlaps — longest-first non-overlap selection.
- 3b: two-pass mask flow (regex inline → ML on partially-masked).
- 3c: placeholder overlap defense + skip patterns.
- 3d: ignore_labels filtering.
- 3e: unmask / unmask_json.
- 3f: mask_obj walker.
- 3g: content-hash cache redesign.
"""

from __future__ import annotations

from anon_proxy.masker import _resolve_overlaps
from anon_proxy.privacy_filter import PIIEntity


def ent(label: str, start: int, end: int, score: float = 0.9) -> PIIEntity:
    """Small helper. `text` is filler — overlap resolution ignores it."""
    return PIIEntity(label=label, text="x" * (end - start), start=start, end=end, score=score)


# ---------------------------------------------------------------------------
# Phase 3a: _resolve_overlaps — longest-first, no-overlap selection.
#
# Algorithm contract:
#   1. Sort candidates by (-length, -score, start, label).
#   2. For each candidate in that order, keep iff it overlaps no already-kept
#      span. Touching at boundaries (e1.end == e2.start) is NOT overlap.
#   3. Return kept spans sorted by start (callers substitute right-to-left).
# ---------------------------------------------------------------------------


class TestResolveOverlapsTrivial:
    def test_empty_returns_empty(self):
        assert _resolve_overlaps([]) == []

    def test_single_entity_kept(self):
        e = ent("PERSON", 0, 5)
        assert _resolve_overlaps([e]) == [e]


class TestResolveOverlapsDisjoint:
    def test_disjoint_spans_all_kept(self):
        a = ent("PERSON", 0, 5)
        b = ent("PERSON", 10, 15)
        c = ent("EMAIL", 20, 30)
        out = _resolve_overlaps([a, b, c])
        assert out == [a, b, c]  # sorted by start

    def test_input_order_does_not_matter(self):
        a = ent("PERSON", 0, 5)
        b = ent("PERSON", 10, 15)
        c = ent("EMAIL", 20, 30)
        assert _resolve_overlaps([c, a, b]) == _resolve_overlaps([a, b, c])

    def test_touching_is_not_overlap(self):
        # b.start == a.end — they share a boundary but do not overlap.
        a = ent("PERSON", 0, 5)
        b = ent("PERSON", 5, 10)
        assert _resolve_overlaps([a, b]) == [a, b]


class TestResolveOverlapsChainedRegression:
    """The bug the redesign fixes: chained replacements that drop a span
    that doesn't conflict with the eventual winner."""

    def test_chain_preserves_non_overlapping_ends(self):
        # A=[0,5], B=[4,10] (longer, overlaps A), C=[7,15] (longer, overlaps B
        # but NOT A). Old algorithm chained: A→B→C, dropping A. New algorithm
        # processes longest first and keeps both A and C.
        a = ent("PERSON", 0, 5)
        b = ent("PERSON", 4, 10)
        c = ent("PERSON", 7, 15)
        out = _resolve_overlaps([a, b, c])
        assert out == [a, c]


class TestResolveOverlapsNestedAndDominating:
    def test_inner_dropped_when_outer_present(self):
        outer = ent("PERSON", 0, 10)
        inner = ent("PERSON", 3, 7)
        assert _resolve_overlaps([outer, inner]) == [outer]

    def test_longer_wins_partial_overlap(self):
        shorter = ent("PERSON", 0, 5)
        longer = ent("PERSON", 3, 12)
        # `longer` (9 chars) beats `shorter` (5 chars).
        assert _resolve_overlaps([shorter, longer]) == [longer]


class TestResolveOverlapsTiebreaks:
    def test_same_span_higher_score_wins(self):
        low = ent("PERSON", 0, 5, score=0.5)
        high = ent("PERSON", 0, 5, score=0.9)
        assert _resolve_overlaps([low, high]) == [high]

    def test_overlapping_same_length_higher_score_wins(self):
        a = ent("PERSON", 0, 5, score=0.5)
        b = ent("PERSON", 2, 7, score=0.9)
        assert _resolve_overlaps([a, b]) == [b]

    def test_same_span_same_score_earliest_start_wins(self):
        # Equal start AND equal length AND equal score → no preference on
        # start (it's the same). Falls through to alphabetical label.
        a = ent("ALPHA", 0, 5, score=0.5)
        b = ent("BETA", 0, 5, score=0.5)
        out = _resolve_overlaps([a, b])
        assert out == [a]  # alphabetical

    def test_overlapping_same_length_same_score_earliest_start_wins(self):
        a = ent("PERSON", 0, 5, score=0.5)
        b = ent("PERSON", 1, 6, score=0.5)
        assert _resolve_overlaps([a, b]) == [a]

    def test_deterministic_under_input_permutation(self):
        # Same input set, two orderings → same output.
        a = ent("PERSON", 0, 5, score=0.5)
        b = ent("PERSON", 2, 7, score=0.5)
        c = ent("EMAIL", 5, 10, score=0.5)
        assert _resolve_overlaps([a, b, c]) == _resolve_overlaps([c, b, a])


class TestResolveOverlapsCrossLabel:
    def test_overlapping_different_labels_longest_wins(self):
        person = ent("PERSON", 0, 5, score=0.9)
        email = ent("EMAIL", 0, 12, score=0.5)
        # EMAIL is longer despite lower score → wins.
        assert _resolve_overlaps([person, email]) == [email]

    def test_identical_span_different_labels_alphabetical(self):
        # Same start, same length, same score → alphabetical label.
        person = ent("PERSON", 0, 5, score=0.5)
        email = ent("EMAIL", 0, 5, score=0.5)
        assert _resolve_overlaps([person, email]) == [email]  # E < P


class TestResolveOverlapsOutputSortedByStart:
    def test_output_sorted_by_start_even_when_input_is_not(self):
        a = ent("PERSON", 30, 40)
        b = ent("PERSON", 0, 10)
        c = ent("PERSON", 15, 25)
        out = _resolve_overlaps([a, b, c])
        assert [e.start for e in out] == [0, 15, 30]
