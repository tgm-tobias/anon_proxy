import json
import os
import re
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class Placeholder:
    label: str
    index: int
    token: str


class PIIStore:
    """In-memory bidirectional map from (label, canonical value) to placeholder tokens.

    Cross-turn consistency: the same entity (modulo casing / whitespace) always
    maps to the same token for the life of this store. The reverse map preserves
    the first-seen original form so un-masking restores the user's casing.

    Thread-safe: a reentrant lock guards the maps and counters so request
    masking can run in worker threads while response unmasking stays inline.
    """

    def __init__(self) -> None:
        self._forward: dict[tuple[str, str], Placeholder] = {}
        self._reverse: dict[str, str] = {}
        self._counters: dict[str, int] = {}
        self._lock = threading.RLock()

    def get_or_create(self, label: str, value: str) -> Placeholder:
        if not value or not value.strip():
            raise ValueError(
                "PIIStore.get_or_create: value must be non-empty after stripping whitespace"
            )
        normalized_label = normalize_label(label)
        key = (normalized_label, _canonical(value))
        with self._lock:
            existing = self._forward.get(key)
            if existing is not None:
                return existing
            index = self._counters.get(normalized_label, 0) + 1
            self._counters[normalized_label] = index
            token = f"<{normalized_label}_{index}>"
            ph = Placeholder(label=normalized_label, index=index, token=token)
            self._forward[key] = ph
            self._reverse[token] = value
            return ph

    def original(self, token: str) -> str | None:
        with self._lock:
            return self._reverse.get(token)

    def tokens(self) -> list[str]:
        with self._lock:
            return list(self._reverse.keys())

    def items(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._reverse.items())

    def __len__(self) -> int:
        with self._lock:
            return len(self._reverse)

    # ---- serialization --------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict.

        The returned dict has two keys:
          ``reverse``  — ``{token: original_value, ...}``
          ``counters`` — ``{label: next_index, ...}``

        The forward map is reconstructed on deserialization.
        """
        with self._lock:
            return {
                "reverse": dict(self._reverse),
                "counters": dict(self._counters),
            }

    @classmethod
    def from_dict(cls, data: dict) -> "PIIStore":
        """Deserialize a dict returned by :meth:`to_dict`.

        Rebuilds the forward lookup from token→original entries, so a
        round-trip preserves all mappings.
        """
        store = cls()
        # The fresh store is not published yet, so direct population does not
        # need to acquire its lock.
        store._reverse = dict(data["reverse"])
        store._counters = dict(data["counters"])
        for token, original in store._reverse.items():
            parsed = _parse_token(token)
            if parsed is None:
                continue
            label, idx = parsed
            key = (label, _canonical(original))
            store._forward[key] = Placeholder(label=label, index=idx, token=token)
        return store

    def save(self, path: str) -> None:
        """Atomically write the store to *path* as JSON."""
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "PIIStore":
        """Read a JSON file written by :meth:`save` and return a new store."""
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: invalid JSON — {e}") from e
        return cls.from_dict(data)


_WHITESPACE = re.compile(r"\s+")


def _canonical(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip().casefold()


def normalize_label(label: str) -> str:
    """Canonical form for a PII label: strip the `private_` prefix the model
    emits, then uppercase. Idempotent.

    Single source of truth — `anon_proxy.config.normalize_label` and the rule
    used by `Masker._ignore_labels` filtering re-export this function.
    """
    trimmed = label[len("private_") :] if label.startswith("private_") else label
    return trimmed.upper()


# Matches placeholder tokens created by PIIStore.get_or_create:
# ``<{LABEL}_{N}>``.  Greedy ``[A-Z0-9_]*`` backtracks far enough for the
# ``_\d+>`` suffix, so labels ending in digits are handled correctly
# (e.g. ``<MY_LABEL_123_1>`` → label=MY_LABEL_123, index=1).
_TOKEN_PARSE_RE = re.compile(r"<([A-Z][A-Z0-9_]*)_(\d+)>")


def _parse_token(token: str) -> tuple[str, int] | None:
    """Split a ``<LABEL_N>`` token into ``(label, index)``.

    Returns ``None`` for strings that don't look like a valid placeholder.
    """
    m = _TOKEN_PARSE_RE.match(token)
    if m is None:
        return None
    return m.group(1), int(m.group(2))
