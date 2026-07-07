import json

import httpx
from starlette.testclient import TestClient

from anon_proxy.masker import Masker
from anon_proxy.metrics import ProxyMetrics
from anon_proxy.server import build_app


class _StubFilter:
    def detect(self, text):
        from anon_proxy.privacy_filter import PIIEntity

        start = text.find("Alice")
        if start == -1:
            return []
        return [
            PIIEntity(
                start=start,
                end=start + 5,
                label="PERSON",
                text="Alice",
                score=0.99,
            )
        ]


def _anthropic_response():
    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello <PERSON_1>"}],
            "usage": {"input_tokens": 10, "output_tokens": 25},
        },
    )


def _client_with_upstream(metrics, masker, handler):
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    app = build_app(masker=masker, metrics=metrics, http_client=http_client)
    return TestClient(app)


def test_successful_request_records_metrics():
    metrics = ProxyMetrics(started_at=0.0)
    masker = Masker(filter=_StubFilter())
    client = _client_with_upstream(metrics, masker, lambda req: _anthropic_response())

    with client:
        resp = client.post(
            "/anthropic/v1/messages",
            headers={"user-agent": "claude-cli/1.2.3 (external, cli)"},
            json={
                "model": "claude-3",
                "messages": [{"role": "user", "content": "Call Alice now"}],
            },
        )

    snap = metrics.snapshot()
    assert resp.status_code == 200
    assert snap["requests_masked_total"] == 1
    assert snap["entities_masked_total"] == 1
    assert snap["last_client"] == "Claude Code"
    assert snap["tokens_out_total"] == 25


def test_masking_error_trips_alarm_and_fails_closed():
    metrics = ProxyMetrics(started_at=0.0)

    class _BoomFilter:
        def detect(self, text):
            raise RuntimeError("detector exploded")

    contacted = {"upstream": False}

    def handler(req):
        contacted["upstream"] = True
        return _anthropic_response()

    client = _client_with_upstream(metrics, Masker(filter=_BoomFilter()), handler)
    with client:
        resp = client.post(
            "/anthropic/v1/messages",
            json={
                "model": "c",
                "messages": [{"role": "user", "content": "hi Alice"}],
            },
        )

    assert resp.status_code == 502
    assert json.loads(resp.text) == {
        "error": "anon-proxy: masking failed; request blocked"
    }
    assert metrics.snapshot()["masking_errors_total"] == 1
    assert contacted["upstream"] is False
