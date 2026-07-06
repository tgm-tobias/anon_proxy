from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy

from anon_proxy.mapping import _parse_token, atomic_write_json, normalize_label


def filter_entries(
    data: dict,
    label: str | None,
    min_len: int | None,
    max_len: int | None,
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    normalized_label = normalize_label(label) if label is not None else None
    for token, value in data["reverse"].items():
        parsed = _parse_token(token)
        if parsed is None:
            continue
        token_label, _index = parsed
        if normalized_label is not None and token_label != normalized_label:
            continue
        value_len = len(value)
        if min_len is not None and value_len < min_len:
            continue
        if max_len is not None and value_len > max_len:
            continue
        rows.append((token, token_label, value))
    return sorted(rows, key=lambda row: row[0])


def purge_tokens(data: dict, tokens: list[str]) -> tuple[dict, list[str], list[str]]:
    new_data = deepcopy(data)
    reverse = new_data["reverse"]
    removed: list[str] = []
    missing: list[str] = []
    for token in tokens:
        if token in reverse:
            del reverse[token]
            removed.append(token)
        else:
            missing.append(token)
    return new_data, removed, missing


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    store_path = args.store or args.global_store or os.environ.get("ANON_PROXY_STORE")
    if not store_path:
        parser.error("--store or ANON_PROXY_STORE is required")

    print(
        "warning: stop anon-proxy before editing the store; "
        "if it is running, restart it to pick up changes.",
        file=sys.stderr,
    )
    data = _load_store(store_path)

    if args.command == "list":
        rows = filter_entries(data, args.label, args.min_len, args.max_len)
        for token, label, value in rows:
            print(f"{token}\t{label}\t{_truncate(value)}\t{len(value)}")
        return 0

    if args.command == "show":
        value = data["reverse"].get(args.token)
        if value is None:
            print(f"not found: {args.token}", file=sys.stderr)
            return 1
        print(value)
        return 0

    if args.command == "purge":
        new_data, removed, missing = purge_tokens(data, args.tokens)
        for token in missing:
            print(f"not found: {token}", file=sys.stderr)
        if removed:
            _backup_then_write(store_path, new_data)
            for token in removed:
                print(f"removed {token}")
        return 1 if missing else 0

    if args.command == "prune":
        _validate_prune_filters(parser, args)
        rows = filter_entries(data, args.label, args.min_len, args.max_len)
        tokens = [token for token, _label, _value in rows]
        new_data, removed, _missing = purge_tokens(data, tokens)
        for token in removed:
            prefix = "would remove" if args.dry_run else "removed"
            print(f"{prefix} {token}")
        if removed and not args.dry_run:
            _backup_then_write(store_path, new_data)
        return 0

    parser.error("missing command")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anon-proxy-store",
        description="Inspect and clean an anon-proxy PII mapping store.",
    )
    parser.add_argument("--store", dest="global_store")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    _add_store_arg(list_parser)
    _add_filter_args(list_parser)

    show_parser = subparsers.add_parser("show")
    _add_store_arg(show_parser)
    show_parser.add_argument("token")

    purge_parser = subparsers.add_parser("purge")
    _add_store_arg(purge_parser)
    purge_parser.add_argument("tokens", nargs="+")

    prune_parser = subparsers.add_parser("prune")
    _add_store_arg(prune_parser)
    _add_filter_args(prune_parser)
    prune_parser.add_argument("--all", action="store_true")
    prune_parser.add_argument("--dry-run", action="store_true")
    return parser


def _add_store_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store")


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--label")
    parser.add_argument("--min-len", type=int)
    parser.add_argument("--max-len", type=int)


def _validate_prune_filters(parser: argparse.ArgumentParser, args) -> None:
    has_filter = (
        args.label is not None or args.min_len is not None or args.max_len is not None
    )
    if not has_filter and not args.all:
        parser.error("prune requires at least one filter or --all")


def _load_store(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _backup_then_write(path: str, new_data: dict) -> None:
    shutil.copyfile(path, path + ".bak")
    atomic_write_json(path, new_data)


def _truncate(value: str, limit: int = 80) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
