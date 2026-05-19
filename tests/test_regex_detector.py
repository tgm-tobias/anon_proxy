"""Tests for RegexDetector.

Specs covered (agreed in Phase 1b):
- Construction compiles all patterns; bad patterns surface as a single
  ValueError listing every failure.
- Empty patterns dict and empty/whitespace text short-circuit to [].
- For each pattern (dict insertion order), emit PIIEntity(score=1.0) for
  each non-zero-length finditer match.
- Labels pass through unchanged (mapping layer normalizes downstream).
- Output ordering is pattern-first, match-second-within-pattern. Overlap
  resolution is NOT this layer's job — overlapping matches from multiple
  patterns are all emitted.
- TODO documented in source for multi-pattern-per-label support.
"""

from __future__ import annotations

import pytest

from anon_proxy.regex_detector import RegexDetector


class TestConstruction:
    def test_empty_patterns_is_valid(self):
        d = RegexDetector({})
        assert len(d) == 0
        assert d.detect("anything") == []

    def test_compiles_all_patterns(self):
        d = RegexDetector({"A": r"\d+", "B": r"[a-z]+"})
        assert len(d) == 2

    def test_single_bad_pattern_raises(self):
        with pytest.raises(ValueError) as exc:
            RegexDetector({"BAD": r"["})
        assert "BAD" in str(exc.value)

    def test_all_bad_patterns_reported_together(self):
        with pytest.raises(ValueError) as exc:
            RegexDetector({"A": r"[", "B": r"(?P<x", "C": r"\d+"})
        msg = str(exc.value)
        # Both bad labels surface; the good one does not.
        assert "A" in msg and "B" in msg
        assert "'C'" not in msg


class TestEmptyInputShortCircuits:
    def test_empty_text_returns_empty(self):
        d = RegexDetector({"DIGIT": r"\d+"})
        assert d.detect("") == []

    def test_whitespace_only_text_returns_empty(self):
        d = RegexDetector({"DIGIT": r"\d+"})
        assert d.detect("  \t\n  ") == []

    def test_no_patterns_returns_empty_regardless_of_text(self):
        d = RegexDetector({})
        assert d.detect("contains 123 digits") == []


class TestSingleMatch:
    def test_single_pattern_single_match(self):
        d = RegexDetector({"DIGIT": r"\d+"})
        [e] = d.detect("call 42 now")
        assert e.label == "DIGIT"
        assert e.text == "42"
        assert (e.start, e.end) == (5, 7)
        assert e.score == 1.0


class TestMultipleMatches:
    def test_single_pattern_multiple_matches(self):
        d = RegexDetector({"DIGIT": r"\d+"})
        out = d.detect("12 then 345 then 6")
        assert [(e.start, e.end, e.text) for e in out] == [
            (0, 2, "12"),
            (8, 11, "345"),
            (17, 18, "6"),
        ]
        assert all(e.label == "DIGIT" for e in out)

    def test_multiple_patterns_ordered_pattern_first_match_second(self):
        # Pattern dict order: DIGIT, WORD. detect output must follow that order,
        # so all DIGIT matches come before any WORD matches.
        patterns = {"DIGIT": r"\d+", "WORD": r"[a-z]+"}
        d = RegexDetector(patterns)
        out = d.detect("abc 12 def 345")
        labels = [e.label for e in out]
        assert labels == ["DIGIT", "DIGIT", "WORD", "WORD"]
        starts = [e.start for e in out]
        # Within each label, matches are in left-to-right order
        assert starts == [4, 11, 0, 7]


class TestZeroLengthSkipped:
    def test_zero_length_match_skipped(self):
        # `\d*` matches at every position including empty matches.
        d = RegexDetector({"DIGITS": r"\d*"})
        out = d.detect("abc 12 def")
        # Only the non-empty match "12" should remain.
        assert [(e.start, e.end, e.text) for e in out] == [(4, 6, "12")]


class TestOverlapsKeptUnresolved:
    def test_overlapping_matches_from_different_patterns_both_emitted(self):
        # Two patterns both match "alice" — the detector must NOT dedupe;
        # overlap resolution is the Masker's responsibility.
        patterns = {"NAME": r"alice", "FIVE_LOWERCASE": r"[a-z]{5}"}
        d = RegexDetector(patterns)
        out = d.detect("hello alice there")
        # NAME: "alice".  FIVE_LOWERCASE: "hello", "alice", "there".
        assert len(out) == 4
        assert {(e.label, e.text) for e in out} == {
            ("NAME", "alice"),
            ("FIVE_LOWERCASE", "hello"),
            ("FIVE_LOWERCASE", "alice"),
            ("FIVE_LOWERCASE", "there"),
        }


class TestLabelPassthrough:
    def test_label_not_normalized(self):
        # The mapping layer normalizes (uppercases, strips 'private_'). This
        # layer must preserve the caller's label verbatim.
        d = RegexDetector({"private_email": r"\w+@\w+", "Phone": r"\d{3}-\d{4}"})
        out = d.detect("me@x and 555-1212")
        labels = {e.label for e in out}
        assert labels == {"private_email", "Phone"}


class TestScore:
    def test_score_is_one(self):
        d = RegexDetector({"X": r"."})
        [e] = d.detect("a")
        assert e.score == 1.0


class TestLen:
    def test_len_matches_pattern_count(self):
        assert len(RegexDetector({})) == 0
        assert len(RegexDetector({"A": "a", "B": "b", "C": "c"})) == 3


class TestEntityShape:
    def test_text_matches_original_slice(self):
        d = RegexDetector({"PHONE": r"\d{3}-\d{4}"})
        text = "call 555-1212 today"
        [e] = d.detect(text)
        assert e.text == text[e.start : e.end]


class TestMultiPatternTODO:
    """We agreed to add a TODO for multi-pattern-per-label support.

    Pin its presence so a future refactor doesn't silently drop the marker.
    """

    def test_todo_present_in_source(self):
        import inspect

        from anon_proxy import regex_detector

        src = inspect.getsource(regex_detector)
        assert "TODO" in src
        assert "multi" in src.lower() or "multiple" in src.lower()
