from __future__ import annotations

from anon_proxy.known_entities import KnownEntityDetector


class TestKnownEntityDetector:
    def test_matches_stored_value_in_code_context(self, store) -> None:
        store.get_or_create("PERSON", "Alice Smith")
        detector = KnownEntityDetector(store)

        entities = detector.detect('git log --author="Alice Smith"')

        assert [(e.label, e.text) for e in entities] == [("PERSON", "Alice Smith")]

    def test_case_insensitive_matches_canonicalization(self, store) -> None:
        store.get_or_create("EMAIL", "Alice@X.com")
        detector = KnownEntityDetector(store)

        entities = detector.detect("send to alice@x.com now")

        assert len(entities) == 1
        assert entities[0].text == "alice@x.com"

    def test_short_values_excluded(self, store) -> None:
        store.get_or_create("PERSON", "la")
        detector = KnownEntityDetector(store)

        assert detector.detect("ls -la && la la la") == []

    def test_word_boundaries(self, store) -> None:
        store.get_or_create("PERSON", "Alice Smith")
        detector = KnownEntityDetector(store)

        assert detector.detect("AliceSmithson") == []

    def test_rebuilds_after_store_growth(self, store) -> None:
        detector = KnownEntityDetector(store)
        assert detector.detect("Bob Jones here") == []

        store.get_or_create("PERSON", "Bob Jones")

        assert len(detector.detect("Bob Jones here")) == 1
