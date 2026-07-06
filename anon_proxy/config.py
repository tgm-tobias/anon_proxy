import json
from dataclasses import dataclass, field
from pathlib import Path

# Re-export so existing `from anon_proxy.config import normalize_label` keeps
# working; the canonical definition lives next to PIIStore which owns the
# placeholder format.
from anon_proxy.mapping import normalize_label  # noqa: F401
from anon_proxy.upstream import UpstreamConfig


_ALLOWED_KEYS = frozenset(
    {
        "patterns",
        "merge_gap",
        "ignore_labels",
        "system_inject",
        "upstreams",
        "default_patterns",
        "canary",
        "min_known_entity_len",
    }
)
_ALLOWED_UPSTREAM_KEYS = frozenset({"base_url", "adapter", "path_prefix", "sse"})
_ALLOWED_ADAPTERS = frozenset({"anthropic", "openai"})


@dataclass(frozen=True)
class Config:
    patterns: dict[str, str] = field(default_factory=dict)
    merge_gap: dict[str, str] = field(default_factory=dict)
    ignore_labels: frozenset[str] = field(default_factory=frozenset)
    system_inject: bool = True
    upstreams: dict[str, UpstreamConfig] = field(default_factory=dict)
    default_patterns: bool = True
    canary: str = "warn"
    min_known_entity_len: int = 6


def load_config(path: str | Path) -> Config:
    """Parse a unified config.json. Shape:

        {
          "patterns":      {"LABEL": "regex", ...},   # optional
          "merge_gap":     {"LABEL": "chars", ...},   # optional
          "ignore_labels": ["LABEL", ...],            # optional
          "system_inject": true,                      # optional, default true
          "default_patterns": true,                   # optional, default true
          "canary": "warn" | "fix" | "off",           # optional, default "warn"
          "min_known_entity_len": 6,                  # optional, 0 disables
          "upstreams": {                              # optional extra providers
            "NAME": {
              "base_url": "https://...",              # required
              "adapter": "anthropic" | "openai",      # optional, default "anthropic"
              "path_prefix": "api/anthropic",         # optional
              "sse": true                             # optional, default true
            }
          }
        }

    Missing top-level keys default to empty. Unknown top-level keys, malformed
    JSON, or wrong-shaped values raise ValueError.
    """
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON — {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level")

    unknown = set(data) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"{path}: unknown top-level keys: {sorted(unknown)!r} "
            f"(allowed: {sorted(_ALLOWED_KEYS)!r})"
        )

    patterns = _str_dict(data.get("patterns", {}), path, "patterns")
    merge_gap = _str_dict(data.get("merge_gap", {}), path, "merge_gap")
    ignore_labels = _str_list_set(data.get("ignore_labels", []), path, "ignore_labels")
    system_inject = _bool(data.get("system_inject", True), path, "system_inject")
    upstreams = _upstreams(data.get("upstreams", {}), path)
    default_patterns = _bool(
        data.get("default_patterns", True), path, "default_patterns"
    )
    canary = _canary(data.get("canary", "warn"), path)
    min_known_entity_len = _nonnegative_int(
        data.get("min_known_entity_len", 6), path, "min_known_entity_len"
    )

    return Config(
        patterns=patterns,
        merge_gap=merge_gap,
        ignore_labels=frozenset(normalize_label(s) for s in ignore_labels),
        system_inject=system_inject,
        upstreams=upstreams,
        default_patterns=default_patterns,
        canary=canary,
        min_known_entity_len=min_known_entity_len,
    )


def _upstreams(value: object, path: str | Path) -> dict[str, UpstreamConfig]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: 'upstreams' must be a JSON object of name -> spec")
    result: dict[str, UpstreamConfig] = {}
    for name, spec in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{path}: 'upstreams' has non-string or empty key: {name!r}"
            )
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: 'upstreams.{name}' must be a JSON object")
        unknown = set(spec) - _ALLOWED_UPSTREAM_KEYS
        if unknown:
            raise ValueError(
                f"{path}: 'upstreams.{name}' has unknown keys: {sorted(unknown)!r} "
                f"(allowed: {sorted(_ALLOWED_UPSTREAM_KEYS)!r})"
            )
        base_url = spec.get("base_url")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError(
                f"{path}: 'upstreams.{name}.base_url' is required and must be a string"
            )
        adapter = spec.get("adapter", "anthropic")
        if adapter not in _ALLOWED_ADAPTERS:
            raise ValueError(
                f"{path}: 'upstreams.{name}.adapter' must be one of "
                f"{sorted(_ALLOWED_ADAPTERS)!r}, got {adapter!r}"
            )
        path_prefix = spec.get("path_prefix", "")
        if not isinstance(path_prefix, str):
            raise ValueError(f"{path}: 'upstreams.{name}.path_prefix' must be a string")
        sse = spec.get("sse", True)
        if not isinstance(sse, bool):
            raise ValueError(f"{path}: 'upstreams.{name}.sse' must be a JSON boolean")
        result[name] = UpstreamConfig(
            name=name,
            base_url=base_url.rstrip("/"),
            path_prefix=path_prefix,
            adapter=adapter,
            sse=sse,
        )
    return result


def _str_dict(value: object, path: str | Path, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {field!r} must be a JSON object of string -> string")
    bad = [
        k for k, v in value.items() if not (isinstance(k, str) and isinstance(v, str))
    ]
    if bad:
        raise ValueError(f"{path}: {field!r} has non-string entries for keys: {bad!r}")
    return dict(value)


def _str_list_set(value: object, path: str | Path, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{path}: {field!r} must be a JSON array of strings")
    bad = [v for v in value if not isinstance(v, str)]
    if bad:
        raise ValueError(f"{path}: {field!r} contains non-string entries: {bad!r}")
    return value


def _bool(value: object, path: str | Path, field: str) -> bool:
    # JSON `true`/`false` parse to Python bool, so accept only that. Reject
    # truthy ints/strings to avoid silently misreading a typo as "on".
    if not isinstance(value, bool):
        raise ValueError(f"{path}: {field!r} must be a JSON boolean (true/false)")
    return value


def _canary(value: object, path: str | Path) -> str:
    if value not in {"warn", "fix", "off"}:
        raise ValueError(f"{path}: 'canary' must be one of ['fix', 'off', 'warn']")
    return str(value)


def _nonnegative_int(value: object, path: str | Path, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path}: {field!r} must be a non-negative integer")
    return value
