"""Replay a captured session through the masking layer to measure new vs old.

Reads a capture.jsonl produced by the proxy with `--capture` and replays each
turn's request body through `mask_request` using the *current* Masker
implementation. For each turn it compares against the captured
`timing_ms.mask_request` from the file — those numbers were recorded by the
proxy at capture time, so as long as you run this on the same machine that
generated the capture, the comparison is apples-to-apples.

Bypasses HTTP entirely. The privacy filter and PIIStore are real, and a single
Masker is reused across the replay so the LRU + block caches build up the way
they would in a live session.

The captured numbers reflect whatever masking code was running when the file
was produced (e.g. the original 256-FIFO content cache, no block cache). The
"new" numbers come from the code in this checkout. Run on the SAME hardware
that captured the file — running on a different box invalidates the diff.

Usage:
  /root/.local/bin/uv run python bench_replay.py [--capture FILE]
                                                 [--limit N]
                                                 [--unmask]
                                                 [--with-baseline]

  # with-baseline also runs an in-process baseline arm (FIFO 256, no block
  # cache) for ablation. Skip it on long captures — it's slow and thermals
  # on a sustained CPU run can make it unreliable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from anon_proxy import Masker
from anon_proxy.adapters import anthropic as anthropic_adapter


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--capture", default="capture.jsonl")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Replay only the first N turns (default: all).",
    )
    p.add_argument(
        "--unmask",
        action="store_true",
        help="Also replay unmask_response on each captured response.",
    )
    p.add_argument(
        "--with-baseline",
        action="store_true",
        help="Also run an in-process baseline arm (FIFO 256, no block "
        "cache) for ablation. Adds a full second pass.",
    )
    args = p.parse_args()

    records = _load(Path(args.capture), args.limit)
    print(f"loaded {len(records)} anthropic /v1/messages records from {args.capture}")
    print()

    # Captured timings from the file: this is the "before" arm — what the
    # proxy actually paid at capture time on the same machine.
    captured = [(r["timing_ms"].get("mask_request", 0.0), 0.0) for r in records]

    optimized = _run_mode(records, "optimized", args.unmask)
    baseline = (
        _run_mode(records, "baseline", args.unmask) if args.with_baseline else None
    )

    print()
    _print_diff(records, captured, optimized, baseline)
    return 0


def _run_mode(records, mode: str, do_unmask: bool):
    print(f"--- mode: {mode} ---")
    masker = Masker()
    if mode == "baseline":
        # Match the captured-session config: small FIFO content cache, no block cache.
        masker._cache_size = 256
        masker._cache.clear()
        # No-op move_to_end => insertion-order eviction (FIFO instead of LRU).
        masker._cache.move_to_end = lambda *a, **k: None  # type: ignore[method-assign]
        masker.mask_obj = lambda obj, walker: walker(obj)  # type: ignore[method-assign]

    print("  warmup...")
    masker.mask("Hello, this is a warmup string for the privacy filter model.")

    print(f"  replaying {len(records)} turns...")
    times: list[tuple[float, float]] = []
    for i, rec in enumerate(records):
        body = rec["request"]["pre_mask"]
        t0 = time.perf_counter()
        anthropic_adapter.mask_request(body, masker)
        mask_ms = (time.perf_counter() - t0) * 1000

        unmask_ms = 0.0
        if do_unmask:
            resp = rec.get("response", {}).get("pre_unmask")
            if isinstance(resp, dict):
                t1 = time.perf_counter()
                anthropic_adapter.unmask_response(resp, masker)
                unmask_ms = (time.perf_counter() - t1) * 1000
        times.append((mask_ms, unmask_ms))
        if (i + 1) % 10 == 0 or i == len(records) - 1:
            print(
                f"    {i + 1:>3}/{len(records)}  cum_mask={sum(t[0] for t in times) / 1000:.1f}s"
            )

    print(f"  detection cache: {len(masker._cache)}/{masker._cache_size}")
    print(f"  block cache:     {len(masker._block_cache)}/{masker._cache_size}")
    print(f"  PIIStore tokens: {len(masker.store)}")
    return times


def _print_diff(records, captured, optimized, baseline=None):
    has_b = baseline is not None
    if has_b:
        header = (
            f"{'turn':>4}  {'msgs':>4}  {'captured_ms':>12}  {'baseline_ms':>12}  "
            f"{'optim_ms':>10}  {'cap/opt':>8}"
        )
    else:
        header = (
            f"{'turn':>4}  {'msgs':>4}  {'captured_ms':>12}  {'optim_ms':>10}  "
            f"{'speedup':>8}  {'saved_ms':>10}"
        )
    print(header)

    cap_tot = b_tot = o_tot = 0.0
    for i, rec in enumerate(records):
        n = len(rec["request"]["pre_mask"].get("messages", []))
        c_m, _ = captured[i]
        o_m, _ = optimized[i]
        cap_tot += c_m
        o_tot += o_m
        speed = (c_m / o_m) if o_m > 0 else float("inf")
        if has_b:
            b_m, _ = baseline[i]
            b_tot += b_m
            print(
                f"{i:>4d}  {n:>4d}  {c_m:>12.1f}  {b_m:>12.1f}  {o_m:>10.1f}  {speed:>7.1f}x"
            )
        else:
            print(
                f"{i:>4d}  {n:>4d}  {c_m:>12.1f}  {o_m:>10.1f}  {speed:>7.1f}x  "
                f"{c_m - o_m:>+10.1f}"
            )

    print()
    print(f"=== AGGREGATE OVER {len(records)} TURNS ===")
    cap_xs = [t[0] for t in captured]
    opt_xs = [t[0] for t in optimized]
    print(
        f"  captured  mask total: {cap_tot / 1000:>8.1f} s   "
        f"mean: {cap_tot / len(records):>8.1f} ms   {_pcts(cap_xs)}"
    )
    if has_b:
        b_xs = [t[0] for t in baseline]
        print(
            f"  baseline  mask total: {b_tot / 1000:>8.1f} s   "
            f"mean: {b_tot / len(records):>8.1f} ms   {_pcts(b_xs)}"
        )
    print(
        f"  optimized mask total: {o_tot / 1000:>8.1f} s   "
        f"mean: {o_tot / len(records):>8.1f} ms   {_pcts(opt_xs)}"
    )
    if o_tot > 0:
        print(
            f"  speedup vs captured:   {cap_tot / o_tot:.2f}x   "
            f"saved: {(cap_tot - o_tot) / 1000:.1f}s "
            f"({100 * (cap_tot - o_tot) / max(cap_tot, 1):.1f}%)"
        )
        if has_b and b_tot > 0:
            print(
                f"  speedup vs baseline:   {b_tot / o_tot:.2f}x   "
                f"saved: {(b_tot - o_tot) / 1000:.1f}s"
            )


def _pcts(xs):
    if not xs:
        return ""
    s = sorted(xs)
    n = len(s)
    return (
        f"p50={s[n // 2]:>7.1f}  p90={s[int(n * 0.9)]:>7.1f}  "
        f"p99={s[min(n - 1, int(n * 0.99))]:>7.1f}  max={s[-1]:>7.1f}"
    )


def _load(path: Path, limit: int | None):
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("provider") != "anthropic":
                continue
            if rec.get("path") != "/v1/messages":
                continue
            if not isinstance(rec.get("request", {}).get("pre_mask"), dict):
                continue
            out.append(rec)
            if limit is not None and len(out) >= limit:
                break
    return out


if __name__ == "__main__":
    raise SystemExit(main())
