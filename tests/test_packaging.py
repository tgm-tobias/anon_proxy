"""Package metadata and console-script contract."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _pyproject() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text())


def test_repo_urls_point_at_byliu_labs():
    urls = _pyproject()["project"]["urls"]
    for key in ("Homepage", "Repository", "Issues"):
        assert "byliu-labs" in urls[key], f"{key} still points at {urls[key]}"


def test_console_scripts_present():
    scripts = _pyproject()["project"]["scripts"]
    assert set(scripts) == {"anon-proxy", "anon-proxy-menubar", "anon-proxy-store"}
    assert scripts["anon-proxy"] == "anon_proxy.cli:main"
