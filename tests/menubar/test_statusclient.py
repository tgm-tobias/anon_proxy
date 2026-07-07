import httpx

from anon_proxy.menubar.statusclient import fetch_status


def test_returns_dict_on_200():
    def fake_get(url, timeout=None):
        return httpx.Response(200, json={"status": "running", "tokens_per_sec": 12.0})

    assert fetch_status("http://x/_status", get=fake_get)["status"] == "running"


def test_returns_none_on_connect_error():
    def fake_get(url, timeout=None):
        raise httpx.ConnectError("refused")

    assert fetch_status("http://x/_status", get=fake_get) is None


def test_returns_none_on_non_200():
    def fake_get(url, timeout=None):
        return httpx.Response(500, text="nope")

    assert fetch_status("http://x/_status", get=fake_get) is None


def test_returns_none_on_bad_json():
    def fake_get(url, timeout=None):
        return httpx.Response(200, text="not json")

    assert fetch_status("http://x/_status", get=fake_get) is None
