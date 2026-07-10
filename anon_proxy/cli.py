"""Top-level ``anon-proxy`` command."""

from __future__ import annotations

import sys
from collections.abc import Callable

from anon_proxy import server


def _serve(argv: list[str]) -> int:
    old_argv = sys.argv
    sys.argv = ["anon-proxy serve", *argv]
    try:
        server.main()
    finally:
        sys.argv = old_argv
    return 0


def _cmd_download_model(args: list[str]) -> int:
    import argparse

    from anon_proxy import model_cache

    parser = argparse.ArgumentParser(prog="anon-proxy download-model")
    parser.add_argument(
        "--backend",
        default="torch",
        help="which weights to fetch (torch, onnx, onnx-q4f16)",
    )
    ns = parser.parse_args(args)
    if model_cache.is_cached(ns.backend):
        print(f"already cached: {model_cache.MODEL_ID} ({ns.backend})")
        return 0
    print(f"downloading {model_cache.MODEL_ID} for backend={ns.backend} ...")
    path = model_cache.download_model(ns.backend, progress=True)
    print(f"done: {path}")
    return 0


def _cmd_setup_client(args: list[str]) -> int:
    import argparse
    from pathlib import Path

    from anon_proxy import client_config

    parser = argparse.ArgumentParser(prog="anon-proxy setup-client")
    parser.add_argument("provider", choices=sorted(client_config.PROVIDERS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--write", metavar="RCFILE")
    ns = parser.parse_args(args)

    url = client_config.base_url_for(ns.provider, ns.host, ns.port)
    line = client_config.env_snippet(ns.provider, url)
    if ns.write:
        client_config.apply_env(ns.provider, url, Path(ns.write).expanduser())
        print(f"wrote {line} to {ns.write}")
    else:
        print(line)
        print(
            f'# then restart your client, or: eval "$(anon-proxy setup-client {ns.provider})"'
        )
    return 0


def _cmd_install_app(args: list[str]) -> int:
    import argparse
    from pathlib import Path

    from anon_proxy import app_bundle

    if sys.platform != "darwin":
        print("install-app is macOS-only.")
        return 1

    parser = argparse.ArgumentParser(prog="anon-proxy install-app")
    parser.add_argument(
        "--dest",
        default=str(Path.home() / "Applications"),
        help="where to write anon-proxy.app (default: ~/Applications)",
    )
    ns = parser.parse_args(args)
    dest = Path(ns.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    app = app_bundle.build_app_bundle(dest, exec_path=app_bundle.menubar_exec_path())
    print(f"installed {app}")
    print(
        "Double-click it, or add it to Login Items, to launch the menu-bar indicator."
    )
    return 0


def _cmd_up(args: list[str]) -> int:
    import argparse

    from anon_proxy import client_config
    from anon_proxy import model_cache

    parser = argparse.ArgumentParser(prog="anon-proxy up")
    parser.add_argument("--backend", default="torch")
    parser.add_argument(
        "--provider", default="claude", choices=sorted(client_config.PROVIDERS)
    )
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    ns = parser.parse_args(args)

    if not ns.skip_download and not model_cache.is_cached(ns.backend):
        print(f"first run: fetching {model_cache.MODEL_ID} ({ns.backend}) ...")
        model_cache.download_model(ns.backend, progress=True)

    url = client_config.base_url_for(ns.provider)
    print("\nPoint your client at the proxy:")
    print(f"  {client_config.env_snippet(ns.provider, url)}\n")

    if ns.no_gui:
        from anon_proxy.server import default_store_path

        serve_argv = [
            "--store",
            str(default_store_path()),
            "--metrics",
            "--backend",
            model_cache.normalize_backend(ns.backend),
        ]
        return _serve(serve_argv)

    from anon_proxy.menubar import app as menubar_app

    menubar_app.main(
        ["--start-proxy", "--backend", model_cache.normalize_backend(ns.backend)]
    )
    return 0


_COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "download-model": _cmd_download_model,
    "setup-client": _cmd_setup_client,
    "install-app": _cmd_install_app,
    "up": _cmd_up,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        return _serve(argv[1:])
    if argv and argv[0] in _COMMANDS:
        return _COMMANDS[argv[0]](argv[1:])
    return _serve(argv)


if __name__ == "__main__":
    raise SystemExit(main())
