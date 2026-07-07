from anon_proxy.menubar.render import format_watch_line, fps_for, render


def _status(**over):
    base = {
        "status": "running",
        "listen_addr": "127.0.0.1:8080",
        "tokens_per_sec": 0.0,
        "requests_masked_total": 0,
        "entities_masked_total": 0,
        "masking_errors_total": 0,
        "tokens_out_total": 0,
        "last_client": None,
        "by_client": {},
        "backend": "mps",
        "uptime_sec": 0.0,
        "store": 0,
    }
    base.update(over)
    return base


def test_fps_scales_and_caps():
    assert fps_for(0) == 1.5
    assert abs(fps_for(280) - (1.5 + 10)) < 1e-9
    assert fps_for(100000) == 12.0


def test_down_when_status_none():
    state = render(None, alarm=False, now=1.0)
    assert state.icon_state == "down"
    assert "not running" in state.tooltip.lower()


def test_idle_when_no_throughput():
    state = render(_status(tokens_per_sec=0.0), alarm=False, now=1.0)
    assert state.icon_state == "idle"


def test_running_when_throughput_positive():
    state = render(
        _status(tokens_per_sec=380.0, last_client="Claude Code"),
        alarm=False,
        now=1.0,
    )
    assert state.icon_state == "running"
    assert state.fps == fps_for(380.0)
    assert "380" in state.title
    assert any("Claude Code" in line for line in state.menu)


def test_alarm_overrides_running():
    state = render(
        _status(tokens_per_sec=380.0, masking_errors_total=2), alarm=True, now=1.0
    )
    assert state.icon_state == "alarm"
    assert any("2" in line and "error" in line.lower() for line in state.menu)


def test_watch_line_is_one_line_string():
    line = format_watch_line(_status(tokens_per_sec=120.0), alarm=False, now=1.0)
    assert "\n" not in line
    assert "120" in line
    assert format_watch_line(None, alarm=False, now=1.0).lower().count("down") == 1
