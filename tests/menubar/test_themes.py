import datetime as dt
from pathlib import Path

from anon_proxy.menubar import themes


def test_holiday_for_winter_and_default():
    assert themes.holiday_for(dt.date(2026, 12, 25)) == "winter"
    assert themes.holiday_for(dt.date(2026, 7, 6)) == "classic"


def test_resolve_theme_auto_and_manual():
    assert themes.resolve_theme("auto", dt.date(2026, 12, 25)) == "winter"
    assert themes.resolve_theme("classic", dt.date(2026, 12, 25)) == "classic"
    assert themes.resolve_theme("no-such-theme", dt.date(2026, 7, 6)) == "classic"


def test_frame_paths_fall_back_to_classic(tmp_path: Path):
    classic = tmp_path / "classic"
    classic.mkdir()
    for frame in themes.FRAMES:
        (classic / f"{frame}.png").write_bytes(b"x")
    spooky = tmp_path / "spooky"
    spooky.mkdir()
    (spooky / "stand.png").write_bytes(b"x")

    paths = themes.frame_paths("spooky", base=tmp_path)
    assert paths["stand"] == spooky / "stand.png"
    assert paths["run1"] == classic / "run1.png"


def test_packaged_classic_frames_exist():
    paths = themes.frame_paths("classic")
    for frame in themes.FRAMES:
        assert paths[frame].exists(), f"missing packaged frame {frame}"
