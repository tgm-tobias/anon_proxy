import json

from anon_proxy.metrics import _MAX_CLIENTS, _OVERFLOW_LABEL, ProxyMetrics


def test_record_request_counts_and_attributes():
    m = ProxyMetrics(started_at=1000.0)
    m.record_request("Claude Code", entities_masked=3, now=1001.0)
    m.record_request("Codex", entities_masked=1, now=1002.0)

    snap = m.snapshot(now=1002.0)

    assert snap["requests_masked_total"] == 2
    assert snap["entities_masked_total"] == 4
    assert snap["last_client"] == "Codex"
    assert snap["last_request_at"] == 1002.0
    assert snap["by_client"]["Claude Code"]["requests"] == 1
    assert snap["by_client"]["Codex"]["requests"] == 1


def test_masking_error_increments_alarm():
    m = ProxyMetrics(started_at=0.0)
    assert m.snapshot(now=0.0)["masking_errors_total"] == 0

    m.record_masking_error()
    m.record_masking_error()

    assert m.snapshot(now=0.0)["masking_errors_total"] == 2


def test_tokens_accumulate_and_attribute():
    m = ProxyMetrics(started_at=0.0)
    m.record_tokens("Claude Code", 100, now=1.0)
    m.record_tokens("Claude Code", 50, now=2.0)

    snap = m.snapshot(now=2.0)

    assert snap["tokens_out_total"] == 150
    assert snap["by_client"]["Claude Code"]["tokens"] == 150


def test_rate_positive_during_burst_and_decays_when_idle():
    m = ProxyMetrics(started_at=0.0)
    for t in (1.0, 1.5, 2.0, 2.5, 3.0):
        m.record_tokens("Claude Code", 100, now=t)

    hot = m.tokens_per_sec(now=3.0)
    cold = m.tokens_per_sec(now=30.0)

    assert hot > 50.0
    assert cold < 1.0


def test_zero_or_negative_tokens_ignored():
    m = ProxyMetrics(started_at=0.0)
    m.record_tokens("x", 0, now=1.0)
    m.record_tokens("x", -5, now=1.0)

    assert m.snapshot(now=1.0)["tokens_out_total"] == 0


def test_snapshot_is_json_safe_and_has_no_content_fields():
    m = ProxyMetrics(started_at=0.0)
    m.record_request("Claude Code", 2, now=1.0)

    snap = m.snapshot(now=1.0)

    json.dumps(snap)
    assert set(snap) == {
        "started_at",
        "uptime_sec",
        "requests_masked_total",
        "entities_masked_total",
        "masking_errors_total",
        "tokens_out_total",
        "tokens_per_sec",
        "last_request_at",
        "last_client",
        "by_client",
    }


def test_by_client_is_capped_under_label_churn():
    m = ProxyMetrics()
    for i in range(_MAX_CLIENTS + 500):
        m.record_request(f"agent-{i}", 1)

    snap = m.snapshot()

    assert len(snap["by_client"]) <= _MAX_CLIENTS + 1
    assert _OVERFLOW_LABEL in snap["by_client"]
    assert snap["requests_masked_total"] == _MAX_CLIENTS + 500


def test_overflow_requests_aggregate_into_other():
    m = ProxyMetrics()
    for i in range(_MAX_CLIENTS):
        m.record_request(f"seed-{i}", 1)
    m.record_request("late-1", 1)
    m.record_request("late-2", 1)
    m.record_tokens("late-3", 10)

    other = m.snapshot()["by_client"][_OVERFLOW_LABEL]

    assert other["requests"] == 2
    assert other["tokens"] == 10


def test_known_client_still_updates_after_cap_reached():
    m = ProxyMetrics()
    m.record_request("claude-code", 1)
    for i in range(_MAX_CLIENTS + 50):
        m.record_request(f"noise-{i}", 1)
    m.record_request("claude-code", 1)

    assert m.snapshot()["by_client"]["claude-code"]["requests"] == 2
