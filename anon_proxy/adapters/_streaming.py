"""Shared SSE-streaming helpers used by both adapters.

The canonical buffer-flush rule for placeholder-aware streaming. Both adapters
import `split_at_last_open` so they emit the same shape; "convergence" is
mechanical rather than convention.
"""

from __future__ import annotations


def split_at_last_open(buf: str) -> tuple[str, str]:
    """Split into (emittable, remainder).

    The remainder is the substring from the last unterminated `<` onward —
    a potentially-incomplete placeholder token that must be held until the
    next chunk arrives. If there's no `<`, or the last `<` has a closing
    `>` after it, the whole buffer is emittable.

    Used by both adapters' streaming pipelines so a placeholder split across
    SSE event boundaries never leaks partially to the client.
    """
    last_open = buf.rfind("<")
    if last_open == -1 or ">" in buf[last_open:]:
        return buf, ""
    return buf[:last_open], buf[last_open:]
