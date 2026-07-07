import httpx

from anon_proxy.menubar.app import AlarmLatch, watch_once


def test_watch_once_running_line():
    def fake_get(url, timeout=None):
        return httpx.Response(
            200,
            json={
                "status": "running",
                "listen_addr": "127.0.0.1:8080",
                "tokens_per_sec": 200.0,
                "requests_masked_total": 3,
                "entities_masked_total": 1,
                "tokens_out_total": 500,
                "masking_errors_total": 0,
                "last_client": "Claude Code",
                "by_client": {},
                "backend": "mps",
                "store": 1,
                "uptime_sec": 5.0,
            },
        )

    line = watch_once("http://x/_status", alarm=False, now=1.0, get=fake_get)
    assert "200" in line and "\n" not in line


def test_watch_once_down_line():
    def fake_get(url, timeout=None):
        raise httpx.ConnectError("refused")

    line = watch_once("http://x/_status", alarm=False, now=1.0, get=fake_get)
    assert "down" in line.lower()


def test_alarm_latch_trips_and_resets():
    latch = AlarmLatch()
    assert latch.update({"masking_errors_total": 0}) is False
    assert latch.update({"masking_errors_total": 1}) is True
    assert latch.update({"masking_errors_total": 1}) is True
    latch.reset()
    assert latch.update({"masking_errors_total": 1}) is False


def test_alarm_latch_ignores_missing_status():
    latch = AlarmLatch()
    assert latch.update(None) is False
