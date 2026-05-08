"""Per-turn capture of mask-eligible request/response pairs to a JSONL file.

Each line is a record describing one mask-eligible turn (request body + response,
both pre- and post-mask, plus per-turn timing breakdown). Used to gather real
workload data for masking-latency analysis. The file contains UNMASKED PII —
treat as sensitive.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


class Capturer:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = asyncio.Lock()

    async def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        async with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
