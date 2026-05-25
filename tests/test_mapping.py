"""Tests for PIIStore (and the public `normalize_label` it owns).

Specs covered (agreed in Phase 2):
- Token format `<{NORMALIZED_LABEL}_{INDEX}>`; index starts at 1.
- Label normalization: strip `private_` prefix, uppercase. Idempotent.
- Value canonicalization (kept as-is): collapse whitespace runs, strip,
  case-fold. Used only as forward-map key; reverse map preserves first-seen.
- get_or_create:
    * miss → new counter increment, new Placeholder, both maps updated
    * hit → existing Placeholder; no counter increment, no reverse update
    * same canonical key under different labels → independent placeholders
    * `private_person` / `PERSON` / `person` all share the PERSON counter
- Empty / whitespace-only value → ValueError.
- tokens()/items() return list copies in insertion order; mutation does not
  affect the store.
- normalize_label is the single source of truth, promoted to mapping.
"""

from __future__ import annotations

import pytest

from anon_proxy.mapping import Placeholder, normalize_label


# ---------------------------------------------------------------------------
# normalize_label — the rule shared with config and masker.
# ---------------------------------------------------------------------------


class TestNormalizeLabel:
    def test_strips_private_prefix(self):
        assert normalize_label("private_person") == "PERSON"

    def test_uppercases(self):
        assert normalize_label("person") == "PERSON"
        assert normalize_label("Person") == "PERSON"

    def test_strip_then_upper(self):
        assert normalize_label("private_email") == "EMAIL"

    def test_does_not_strip_non_private_prefix(self):
        assert normalize_label("publicperson") == "PUBLICPERSON"
        assert normalize_label("private") == "PRIVATE"  # no underscore → no strip

    def test_idempotent(self):
        for inp in ["person", "PERSON", "private_person", "private_EMAIL"]:
            once = normalize_label(inp)
            assert normalize_label(once) == once


class TestNormalizeLabelIsTheSourceOfTruth:
    """Phase 2.5 lite — the rule must be shared across config and (when used)
    masker, so user-supplied ignore_labels match what PIIStore produces."""

    def test_config_normalize_label_is_same_function(self):
        from anon_proxy.config import normalize_label as config_normalize
        from anon_proxy.mapping import normalize_label as mapping_normalize

        assert config_normalize is mapping_normalize

    def test_masker_uses_the_same_rule(self):
        # masker imports normalize_label; whichever module it imports from
        # must produce the same output as mapping.normalize_label.
        from anon_proxy import masker

        for inp in ["private_person", "PERSON", "person", "private_email"]:
            assert masker.normalize_label(inp) == normalize_label(inp)


# ---------------------------------------------------------------------------
# get_or_create — happy paths.
# ---------------------------------------------------------------------------


class TestGetOrCreateBasics:
    def test_first_call_creates_placeholder_with_index_one(self, store):
        ph = store.get_or_create("PERSON", "Alice")
        assert ph == Placeholder(label="PERSON", index=1, token="<PERSON_1>")

    def test_token_format(self, store):
        ph = store.get_or_create("EMAIL", "a@b.com")
        assert ph.token == "<EMAIL_1>"

    def test_counter_increments_per_label(self, store):
        a = store.get_or_create("PERSON", "Alice")
        b = store.get_or_create("PERSON", "Bob")
        assert (a.index, b.index) == (1, 2)
        assert (a.token, b.token) == ("<PERSON_1>", "<PERSON_2>")


class TestRepeatedReuse:
    def test_same_canonical_returns_same_placeholder_no_counter_bump(self, store):
        first = store.get_or_create("PERSON", "Alice")
        second = store.get_or_create("PERSON", "Alice")
        assert first is second  # exact same object — no allocation
        assert len(store) == 1

    def test_case_variants_share_placeholder(self, store):
        a = store.get_or_create("PERSON", "Alice Smith")
        b = store.get_or_create("PERSON", "alice smith")
        c = store.get_or_create("PERSON", "ALICE SMITH")
        assert a is b is c
        assert len(store) == 1

    def test_whitespace_variants_share_placeholder(self, store):
        a = store.get_or_create("PERSON", "Alice Smith")
        b = store.get_or_create("PERSON", "  Alice   Smith  ")
        c = store.get_or_create("PERSON", "Alice\tSmith")
        assert a is b is c
        assert len(store) == 1

    def test_first_seen_original_wins_in_reverse_map(self, store):
        store.get_or_create("PERSON", "Alice")
        store.get_or_create("PERSON", "alice")
        store.get_or_create("PERSON", "ALICE")
        assert store.original("<PERSON_1>") == "Alice"


# ---------------------------------------------------------------------------
# Label normalization integrated with the store.
# ---------------------------------------------------------------------------


class TestLabelNormalizationInStore:
    def test_private_prefix_label_strips_and_uppercases(self, store):
        ph = store.get_or_create("private_person", "Alice")
        assert ph.label == "PERSON"
        assert ph.token == "<PERSON_1>"

    def test_lower_and_upper_share_counter(self, store):
        a = store.get_or_create("PERSON", "Alice")
        b = store.get_or_create("person", "Bob")
        c = store.get_or_create("private_person", "Carol")
        assert (a.index, b.index, c.index) == (1, 2, 3)

    def test_different_labels_use_independent_counters(self, store):
        p = store.get_or_create("PERSON", "Alice")
        e = store.get_or_create("EMAIL", "a@b.com")
        assert (p.index, e.index) == (1, 1)

    def test_same_value_under_different_labels_get_independent_placeholders(
        self, store
    ):
        # Edge case: "alice" tagged as both PERSON and ORG.
        p = store.get_or_create("PERSON", "alice")
        o = store.get_or_create("ORGANIZATION", "alice")
        assert p is not o
        assert p.token != o.token
        assert len(store) == 2


# ---------------------------------------------------------------------------
# Empty / whitespace-only values must be rejected.
# ---------------------------------------------------------------------------


class TestRejectEmpty:
    def test_empty_string_raises(self, store):
        with pytest.raises(ValueError):
            store.get_or_create("PERSON", "")

    def test_whitespace_only_raises(self, store):
        with pytest.raises(ValueError):
            store.get_or_create("PERSON", "   \t\n")

    def test_failed_call_does_not_advance_counter(self, store):
        with pytest.raises(ValueError):
            store.get_or_create("PERSON", "")
        ph = store.get_or_create("PERSON", "Alice")
        assert ph.index == 1  # counter never bumped on the failed call


# ---------------------------------------------------------------------------
# Reverse lookups and iteration.
# ---------------------------------------------------------------------------


class TestOriginal:
    def test_known_token_returns_first_seen_original(self, store):
        store.get_or_create("PERSON", "Alice")
        assert store.original("<PERSON_1>") == "Alice"

    def test_unknown_token_returns_none(self, store):
        assert store.original("<PERSON_999>") is None
        assert store.original("not a token") is None


class TestTokensAndItems:
    def test_tokens_in_insertion_order(self, store):
        store.get_or_create("PERSON", "Alice")
        store.get_or_create("EMAIL", "a@b.com")
        store.get_or_create("PERSON", "Bob")
        assert store.tokens() == ["<PERSON_1>", "<EMAIL_1>", "<PERSON_2>"]

    def test_items_in_insertion_order(self, store):
        store.get_or_create("PERSON", "Alice")
        store.get_or_create("EMAIL", "a@b.com")
        assert store.items() == [
            ("<PERSON_1>", "Alice"),
            ("<EMAIL_1>", "a@b.com"),
        ]

    def test_tokens_returns_copy(self, store):
        store.get_or_create("PERSON", "Alice")
        snapshot = store.tokens()
        snapshot.append("<MUTATED>")
        assert store.tokens() == ["<PERSON_1>"]

    def test_items_returns_copy(self, store):
        store.get_or_create("PERSON", "Alice")
        snapshot = store.items()
        snapshot.clear()
        assert store.items() == [("<PERSON_1>", "Alice")]


class TestLen:
    def test_starts_at_zero(self, store):
        assert len(store) == 0

    def test_matches_distinct_placeholder_count(self, store):
        store.get_or_create("PERSON", "Alice")
        store.get_or_create("PERSON", "alice")  # same canonical → no growth
        store.get_or_create("EMAIL", "a@b.com")
        assert len(store) == 2


# ---------------------------------------------------------------------------
# Placeholder dataclass invariants.
# ---------------------------------------------------------------------------


class TestPlaceholder:
    def test_is_frozen(self):
        ph = Placeholder(label="PERSON", index=1, token="<PERSON_1>")
        with pytest.raises(Exception):
            ph.index = 2  # frozen dataclasses raise FrozenInstanceError (a TypeError)

    def test_token_field_matches_label_and_index(self, store):
        # The store always constructs Placeholders with consistent fields.
        ph = store.get_or_create("PERSON", "Alice")
        assert ph.token == f"<{ph.label}_{ph.index}>"
