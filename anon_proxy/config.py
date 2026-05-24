import json
from dataclasses import dataclass, field
from pathlib import Path

# Re-export so existing `from anon_proxy.config import normalize_label` keeps
# working; the canonical definition lives next to PIIStore which owns the
# placeholder format.
from anon_proxy.mapping import normalize_label  # noqa: F401


_ALLOWED_KEYS = frozenset({"patterns", "merge_gap", "ignore_labels", "system_inject"})


@dataclass(frozen=True)
class Config:
    patterns: dict[str, str] = field(default_factory=dict)
    merge_gap: dict[str, str] = field(default_factory=dict)
    ignore_labels: frozenset[str] = field(default_factory=frozenset)
    system_inject: bool = True


def load_config(path: str | Path) -> Config:
    """Parse a unified config.json. Shape:

        {
          "patterns":      {"LABEL": "regex", ...},   # optional
          "merge_gap":     {"LABEL": "chars", ...},   # optional
          "ignore_labels": ["LABEL", ...],            # optional
          "system_inject": true                       # optional, default true
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

    return Config(
        patterns=patterns,
        merge_gap=merge_gap,
        ignore_labels=frozenset(normalize_label(s) for s in ignore_labels),
        system_inject=system_inject,
    )


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
