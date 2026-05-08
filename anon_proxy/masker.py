import contextlib
import contextvars
import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import Any, Callable, Protocol

from anon_proxy.mapping import PIIStore
from anon_proxy.privacy_filter import PIIEntity, PrivacyFilter


_TELEMETRY: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "anon_proxy_masker_telemetry", default=None,
)


@contextlib.contextmanager
def telemetry_scope():
    """Collect per-call masker telemetry into a fresh list for the current task.

    Each entry: {"op": "mask"|"unmask"|"unmask_json", "chars": int, "ms": float,
    "cache_hit": bool (mask only), "skipped": bool (mask, optional)}.
    Safe under concurrent asyncio tasks — each task gets its own list via contextvars.
    """
    record: list = []
    token = _TELEMETRY.set(record)
    try:
        yield record
    finally:
        _TELEMETRY.reset(token)


class Detector(Protocol):
    def detect(self, text: str) -> list[PIIEntity]: ...


# Patterns for content that should never be masked (non-user PII content)
_SKIP_MASK_PATTERNS = [
    # Claude Code system-reminder blocks - contain tool definitions, skills list, etc.
    re.compile(r'^\s*<system-reminder>', re.MULTILINE),
    # Tool result blocks that are purely structural (e.g., file listings, tool outputs)
    # These can be extended as needed
]


class Masker:
    """Composes PrivacyFilter + PIIStore to mask outgoing text and unmask LLM replies.

    One Masker instance per conversation: the store accumulates entities across
    turns so the same PII always gets the same placeholder.

    `extra_detectors` is a list of objects with a `detect(text) -> list[PIIEntity]`
    method whose spans are merged into the primary filter's output. Overlapping
    spans from different detectors are resolved by preferring the longer span.

    Performance optimizations:
    - LRU caches detection results by content hash to avoid re-scanning identical text
    - LRU caches masked block-shaped objects by content hash so repeated message
      blocks (the common shape of conversation history) skip re-walking entirely
    - Skips masking for known non-PII patterns (e.g., system-reminders)
    - Early-return if content already contains only placeholders (no new PII)
    """

    def __init__(
        self,
        filter: PrivacyFilter | None = None,
        store: PIIStore | None = None,
        extra_detectors: list[Detector] | None = None,
        skip_patterns: list[re.Pattern] | None = None,
        cache_size: int = 4096,
    ) -> None:
        self._filter = filter or PrivacyFilter()
        self._store = store or PIIStore()
        self._extra: list[Detector] = list(extra_detectors or [])
        self._skip_patterns = skip_patterns or _SKIP_MASK_PATTERNS
        self._cache_size = cache_size
        # LRU cache: content_hash -> (entities, masked_text)
        self._cache: OrderedDict[str, tuple[list[PIIEntity], str]] = OrderedDict()
        # LRU cache: block_hash -> already-masked block-shaped object
        self._block_cache: OrderedDict[str, Any] = OrderedDict()

    @property
    def store(self) -> PIIStore:
        return self._store

    def mask(self, text: str) -> str:
        record = _TELEMETRY.get()
        t0 = time.perf_counter() if record is not None else 0.0

        # Fast path: check if this text matches any skip pattern
        for pattern in self._skip_patterns:
            if pattern.search(text):
                if record is not None:
                    record.append({
                        "op": "mask", "chars": len(text),
                        "ms": (time.perf_counter() - t0) * 1000,
                        "cache_hit": False, "skipped": True,
                    })
                return text  # Skip masking entirely

        # Check cache
        content_hash = _hash_content(text)
        if cached := self._cache.get(content_hash):
            self._cache.move_to_end(content_hash)
            if record is not None:
                record.append({
                    "op": "mask", "chars": len(text),
                    "ms": (time.perf_counter() - t0) * 1000,
                    "cache_hit": True,
                })
            return cached[1]

        # Detect entities
        entities: list[PIIEntity] = list(self._filter.detect(text))
        for detector in self._extra:
            entities.extend(detector.detect(text))
        entities = _resolve_overlaps(entities)

        # Early return if no entities found
        if not entities:
            self._cache_result(content_hash, [], text)
            if record is not None:
                record.append({
                    "op": "mask", "chars": len(text),
                    "ms": (time.perf_counter() - t0) * 1000,
                    "cache_hit": False,
                })
            return text

        # Replace right-to-left so earlier spans' offsets stay valid.
        masked = text
        for e in sorted(entities, key=lambda x: x.start, reverse=True):
            token = self._store.get_or_create(e.label, e.text).token
            masked = masked[: e.start] + token + masked[e.end :]

        self._cache_result(content_hash, entities, masked)
        if record is not None:
            record.append({
                "op": "mask", "chars": len(text),
                "ms": (time.perf_counter() - t0) * 1000,
                "cache_hit": False,
            })
        return masked

    def _cache_result(self, content_hash: str, entities: list[PIIEntity], masked: str) -> None:
        """Cache a detection result with LRU eviction."""
        self._cache[content_hash] = (entities, masked)
        self._cache.move_to_end(content_hash)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def mask_obj(self, obj: Any, walker: Callable[[Any], Any]) -> Any:
        """Mask a JSON-shaped value (typically a message or content block) with
        a content-hash cache so repeated objects skip re-walking entirely.

        `walker` is the function that produces a freshly-masked version of `obj`
        from scratch — invoked only on a cache miss. The returned masked object
        is cached and shared across callers; do not mutate it.

        Designed for conversation history: at turn N+1, `messages[0..N-1]` are
        byte-identical to turn N (and already contain only placeholder tokens),
        so a hash hit short-circuits the entire recursive walk.
        """
        try:
            key = _hash_obj(obj)
        except (TypeError, ValueError):
            return walker(obj)
        record = _TELEMETRY.get()
        t0 = time.perf_counter() if record is not None else 0.0
        cached = self._block_cache.get(key)
        if cached is not None:
            self._block_cache.move_to_end(key)
            if record is not None:
                record.append({
                    "op": "mask_obj",
                    "ms": (time.perf_counter() - t0) * 1000,
                    "cache_hit": True,
                })
            return cached
        result = walker(obj)
        self._block_cache[key] = result
        self._block_cache.move_to_end(key)
        while len(self._block_cache) > self._cache_size:
            self._block_cache.popitem(last=False)
        if record is not None:
            record.append({
                "op": "mask_obj",
                "ms": (time.perf_counter() - t0) * 1000,
                "cache_hit": False,
            })
        return result

    def unmask(self, text: str) -> str:
        record = _TELEMETRY.get()
        t0 = time.perf_counter() if record is not None else 0.0
        result = self._sub(text, lambda s: s)
        if record is not None:
            record.append({
                "op": "unmask", "chars": len(text),
                "ms": (time.perf_counter() - t0) * 1000,
            })
        return result

    def unmask_json(self, text: str) -> str:
        """Unmask tokens sitting inside a JSON string context.

        Replacements are JSON-escaped so an original containing `"`, `\\`, or
        control chars doesn't break the surrounding JSON. Use this for raw
        JSON fragments like Anthropic's `input_json_delta.partial_json` where
        the unmasked text flows through an unparsed string.
        """
        record = _TELEMETRY.get()
        t0 = time.perf_counter() if record is not None else 0.0
        result = self._sub(text, lambda s: json.dumps(s)[1:-1])
        if record is not None:
            record.append({
                "op": "unmask_json", "chars": len(text),
                "ms": (time.perf_counter() - t0) * 1000,
            })
        return result

    def _sub(self, text: str, transform: Callable[[str], str]) -> str:
        """Substitute placeholder tokens with their original values."""
        tokens = self._store.tokens()
        if not tokens:
            return text
        # Longest-first so "<PERSON_1>" can't shadow "<PERSON_10>".
        pattern = re.compile(
            "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))
        )

        def repl(m: re.Match[str]) -> str:
            original = self._store.original(m.group(0))
            return transform(original) if original is not None else m.group(0)

        return pattern.sub(repl, text)


def _resolve_overlaps(entities: list[PIIEntity]) -> list[PIIEntity]:
    """Keep a non-overlapping subset of spans.

    Greedy: sort by (start, -length, -score) so earlier and longer spans land first.
    Walk left-to-right; when a span overlaps the last kept, replace only if the
    new one is strictly longer (ties: higher score wins). Touching spans at
    `prev.end == next.start` do not overlap.
    """
    if not entities:
        return entities
    ordered = sorted(
        entities,
        key=lambda e: (e.start, -(e.end - e.start), -e.score, e.label),
    )
    kept: list[PIIEntity] = []
    for e in ordered:
        if kept and e.start < kept[-1].end:
            prev = kept[-1]
            prev_len = prev.end - prev.start
            cur_len = e.end - e.start
            if cur_len > prev_len or (cur_len == prev_len and e.score > prev.score):
                kept[-1] = e
            continue
        kept.append(e)
    return kept


def _hash_content(text: str) -> str:
    """Hash content for caching detection results.

    Uses SHA256 truncated to 12 chars (collision-resistant enough for cache keys,
    compact enough to be memory-efficient).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _hash_obj(obj: Any) -> str:
    """Hash a JSON-shaped value by its canonical JSON serialization."""
    serialized = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
