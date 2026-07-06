## What

Adds `--backend onnx`: runs the pre-quantized `model_q4f16.onnx` export that
the `openai/privacy-filter` repo **already ships** through ONNX Runtime,
instead of the torch pipeline. Also removes the `--backend mlx` /
`--mlx-weights-cache` flags, which advertised a backend that never existed and
crashed the server at startup (**Fixes #15**).

torch stays the default and is byte-identical — `detect()` still calls the
pipeline once per chunk at `chunk_size=1500`.

## Why it's worth it

12-turn agent replay, each request carrying the full history (the dominant
real shape), on a CPU-only Apple Silicon laptop:

| backend | cold start | warm median | warm p95 | 12-turn total |
|---------|-----------:|------------:|---------:|--------------:|
| torch (default) | 7.3 s | 9.2 s | 10.0 s | 109.2 s |
| **onnx q4f16** | 1.8 s | **0.67 s** | 0.86 s | **9.6 s** |

**≈14× faster on warm turns, ≈11× lower total latency**, largest exactly where
torch hurts most (CPU-only boxes). Reproduce:
`ANON_PROXY_LIVE_TESTS=1 uv run --extra onnx python scripts/bench_masking.py`

## Correctness gate

A quantized graph that silently drops an entity is a privacy regression, not a
perf win. `tests/test_backend_parity.py` (opt-in, real model) asserts the
invariant that matters: **the onnx backend masks every character the torch
backend masks** across a golden set — names, emails, phones, addresses, a
chunk-boundary straddle, CJK, code, and a PII-free line. Over-masking is
allowed and logged; a single missed character fails the gate.

Run on this branch (`ANON_PROXY_LIVE_TESTS=1 uv run pytest tests/test_backend_parity.py`):

```
tests/test_backend_parity.py::test_onnx_covers_every_char_torch_masks PASSED
tests/test_backend_parity.py::test_onnx_finds_the_obvious_pii PASSED
2 passed in 38.22s
```

The BIOES token-aggregation logic is also unit-tested offline, so CI covers the
onnx code path without downloading the model.

## Design notes

- **onnx deps are opt-in** (`uv sync --extra onnx`). We call `onnxruntime`
  directly rather than via `optimum`, which caps `transformers < 4.58` and so
  conflicts with this project's `transformers >= 5`. A small pipeline-shaped
  classifier does the same BIOES aggregation the HF `simple` strategy would.
- `id2label` is read straight from `config.json` — no `AutoConfig`, no
  `trust_remote_code`, no custom-architecture resolution.
- First `--backend onnx` run downloads the ~0.77 GB q4f16 graph (+ its weights
  sidecar); subsequent runs are cached.

## Testing

- `uv run pytest -q` → **346 passed, 2 skipped** (parity skipped without the
  live flag).
- `ruff check .` and `ruff format --check .` → clean.
- Live parity gate + benchmark run against the real model (numbers above).

## Base

Cut from `main` (this repo). Standalone — no other feature work rides along.
