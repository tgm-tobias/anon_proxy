from __future__ import annotations

from anon_proxy.masker import Masker
from anon_proxy.registry import MaskerRegistry, client_id


class TestClientId:
    def test_x_api_key_hashed(self):
        cid = client_id({"x-api-key": "sk-ant-123"})

        assert cid and len(cid) == 16 and "sk-ant" not in cid

    def test_authorization_fallback(self):
        assert client_id({"authorization": "Bearer tok"}) is not None

    def test_no_credential_is_none(self):
        assert client_id({"content-type": "application/json"}) is None

    def test_different_keys_different_ids(self):
        assert client_id({"x-api-key": "a"}) != client_id({"x-api-key": "b"})


class TestMaskerRegistry:
    def test_same_client_same_masker(self, make_filter):
        reg = MaskerRegistry(
            lambda store: Masker(filter=make_filter(), store=store), store_dir=None
        )

        assert reg.get("abc") is reg.get("abc")

    def test_clients_are_isolated(self, make_filter):
        reg = MaskerRegistry(
            lambda store: Masker(filter=make_filter(), store=store), store_dir=None
        )
        m_a = reg.get("aaaa")
        m_b = reg.get("bbbb")
        m_a.store.get_or_create("PERSON", "Alice")

        assert m_b.unmask("hi <PERSON_1>") == "hi <PERSON_1>"

    def test_store_dir_roundtrip(self, tmp_path, make_filter):
        reg = MaskerRegistry(
            lambda store: Masker(filter=make_filter(), store=store),
            store_dir=str(tmp_path),
        )
        reg.get("cafe").store.get_or_create("PERSON", "Alice")
        reg.get("cafe").store.save(reg.store_path("cafe"))

        reg2 = MaskerRegistry(
            lambda store: Masker(filter=make_filter(), store=store),
            store_dir=str(tmp_path),
        )

        assert reg2.get("cafe").store.original("<PERSON_1>") == "Alice"
