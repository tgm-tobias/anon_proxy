"""Per-client Masker namespacing for shared multi-user deployments.

Identity is a hash of the client's upstream credential. The hash is used for
filenames and logs so credential material is never stored by the registry.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Callable, Mapping

from anon_proxy.mapping import PIIStore
from anon_proxy.masker import Masker

_CRED_HEADERS = ("x-api-key", "authorization")


def client_id(headers: Mapping[str, str]) -> str | None:
    lowered = {k.lower(): v for k, v in headers.items()}
    for header in _CRED_HEADERS:
        value = lowered.get(header)
        if value:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return None


class MaskerRegistry:
    def __init__(
        self,
        make_masker: Callable[[PIIStore], Masker],
        store_dir: str | None,
    ) -> None:
        self._make_masker = make_masker
        self._store_dir = store_dir
        self._maskers: dict[str, Masker] = {}
        self._lock = threading.Lock()
        if store_dir:
            # Store files map placeholders to raw PII; keep the directory
            # owner-only. Does not tighten a pre-existing looser directory.
            os.makedirs(store_dir, mode=0o700, exist_ok=True)

    def store_path(self, cid: str) -> str | None:
        if self._store_dir is None:
            return None
        return os.path.join(self._store_dir, f"{cid}.json")

    def get(self, cid: str) -> Masker:
        with self._lock:
            masker = self._maskers.get(cid)
            if masker is not None:
                return masker
            store = PIIStore()
            path = self.store_path(cid)
            if path and os.path.exists(path):
                store = PIIStore.load(path)
            masker = self._make_masker(store)
            self._maskers[cid] = masker
            return masker
