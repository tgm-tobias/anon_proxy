"""In-memory, thread-safe proxy activity metrics.

Holds only counts and agent labels: never request content, PII originals,
placeholder mappings, or auth headers.
"""

from __future__ import annotations

import math
import threading
import time

_TAU = 3.0


class ProxyMetrics:
    def __init__(self, started_at: float | None = None) -> None:
        self.started_at = time.time() if started_at is None else started_at
        self.requests_masked_total = 0
        self.entities_masked_total = 0
        self.masking_errors_total = 0
        self.tokens_out_total = 0
        self.last_request_at: float | None = None
        self.last_client: str | None = None
        self.by_client: dict[str, dict[str, int]] = {}
        self._rate = 0.0
        self._last_token_ts: float | None = None
        self._lock = threading.Lock()

    def _client(self, label: str) -> dict[str, int]:
        client = self.by_client.get(label)
        if client is None:
            client = {"requests": 0, "tokens": 0}
            self.by_client[label] = client
        return client

    def record_request(
        self, client_label: str, entities_masked: int, now: float | None = None
    ) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self.requests_masked_total += 1
            self.entities_masked_total += max(0, entities_masked)
            self.last_request_at = now
            self.last_client = client_label
            self._client(client_label)["requests"] += 1

    def record_masking_error(self, now: float | None = None) -> None:
        with self._lock:
            self.masking_errors_total += 1

    def record_tokens(
        self, client_label: str, n: int, now: float | None = None
    ) -> None:
        if n <= 0:
            return
        now = time.time() if now is None else now
        with self._lock:
            self.tokens_out_total += n
            self._client(client_label)["tokens"] += n
            if self._last_token_ts is None:
                self._rate = float(n)
            else:
                dt = max(now - self._last_token_ts, 1e-3)
                inst = n / dt
                alpha = 1.0 - math.exp(-dt / _TAU)
                self._rate = alpha * inst + (1.0 - alpha) * self._rate
            self._last_token_ts = now

    def _decayed_rate(self, now: float) -> float:
        if self._last_token_ts is None:
            return 0.0
        dt = max(now - self._last_token_ts, 0.0)
        return self._rate * math.exp(-dt / _TAU)

    def tokens_per_sec(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        with self._lock:
            return self._decayed_rate(now)

    def snapshot(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            return {
                "started_at": self.started_at,
                "uptime_sec": max(0.0, now - self.started_at),
                "requests_masked_total": self.requests_masked_total,
                "entities_masked_total": self.entities_masked_total,
                "masking_errors_total": self.masking_errors_total,
                "tokens_out_total": self.tokens_out_total,
                "tokens_per_sec": round(self._decayed_rate(now), 1),
                "last_request_at": self.last_request_at,
                "last_client": self.last_client,
                "by_client": {k: dict(v) for k, v in self.by_client.items()},
            }
