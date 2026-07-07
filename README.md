<img width="108" height="112" alt="anon" src="https://github.com/user-attachments/assets/6609f7ff-3e0b-458d-ac20-2f1b0b95ae62" />

# anon-proxy

**Use Claude Code, ChatGPT, and other LLM APIs on sensitive data without sending raw PII to the cloud.** A local privacy proxy that masks personal information *before* requests leave your device and unmasks it in responses. The [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) model runs entirely on your machine — names, emails, phone numbers, and addresses never reach Anthropic, OpenAI, or any other upstream API.

[![CI](https://github.com/KevinXuxuxu/anon_proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/KevinXuxuxu/anon_proxy/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Works with Claude Code](https://img.shields.io/badge/works%20with-Claude%20Code-orange)

```
your client  →  anon-proxy (mask/unmask)  →  api.anthropic.com | api.openai.com | ...
```

---

## Who is this for?

- **Engineers in regulated industries** (healthcare, legal, finance) whose employer policy or compliance regime (HIPAA, GDPR, SOC 2) blocks raw customer data from being sent to third-party LLM APIs.
- **Developers using Claude Code or OpenAI SDKs on production data** — debugging customer support tickets, summarizing user emails, analyzing logs that contain real names and identifiers.
- **Privacy-conscious users** who want LLM productivity without the data exhaust of pasting personal information into a cloud model.
- **Self-hosters** building on top of LLM APIs who need a redaction layer between their app and the provider.

If you've ever caught yourself manually find-and-replacing real names in a prompt, this is for you.

---

## Why not just use X?

| Tool | What it does | Why it's not the same |
|---|---|---|
| **Microsoft Presidio** | Regex + spaCy NER for PII detection | Library, not a proxy. You still have to wire it into every LLM call yourself. No stable token mapping across turns. |
| **AWS Comprehend / GCP DLP** | Cloud-based PII detection APIs | Sends your data to *another* cloud provider. Defeats the purpose if your goal is "nothing leaves the box." |
| **LiteLLM proxy** | Multi-provider LLM routing | Doesn't redact. Solves a different problem (routing/cost) entirely. |
| **Prompting "please don't log my data"** | 🙏 | Not a security model. |
| **anon-proxy** | Local ML detector + transparent proxy with stable per-session placeholders | Drop-in `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` swap. No code changes in the client. PII gets the same placeholder every turn so the model stays coherent. |

---

## Multi-provider support

The proxy uses **sub-routing** to support multiple API providers:

```
/{provider}/{api-path}  →  {provider-base-url}/{api-path}
```

Examples:
- `/anthropic/v1/messages` → `https://api.anthropic.com/v1/messages`
- `/openai/v1/chat/completions` → `https://api.openai.com/v1/chat/completions`
- `/zai/v1/messages` → `https://api.z.ai/api/anthropic/v1/messages`

Built-in providers: `anthropic`, `openai`, `zai`. Add custom providers with `--extra-upstream`.

---

## Quick demo

```bash
# test the PII detector interactively
uv run python test_filter.py "Alice Smith called from 555-867-5309, email alice@company.com"
```
```
[private_person:Alice] [private_person:Smith] called from [private_phone:555-867-5309], email [private_email:alice@company.com]

  private_person 'Alice'                        score=1.000  offset=0-5
  private_person 'Smith'                        score=1.000  offset=6-11
  private_phone '555-867-5309'                 score=1.000  offset=24-36
  private_email 'alice@company.com'            score=1.000  offset=44-61
```

```bash
# interactive chat through the mask/unmask layer (needs ANTHROPIC_API_KEY)
uv run python test_mask.py
```
```
you[1]> My name is Alice Smith. Summarize this note from bob@acme.com.
  sending -> My name is <PERSON_1>. Summarize this note from <EMAIL_1>.

claude[1]> Sure <PERSON_1>, here's the summary of the note from <EMAIL_1>: ...
  rendered -> Sure Alice Smith, here's the summary of the note from bob@acme.com: ...
```

---

## Prerequisites

- Python ≥ 3.10 (use [uv](https://docs.astral.sh/uv/))
- CUDA GPU recommended (≥4 GB VRAM); CPU works but is slower
- Apple Silicon (M1/M2/M3/M4) supported via the MPS backend
- `ANTHROPIC_API_KEY` for `test_mask.py`; the proxy itself forwards client auth — no key needed on the server

```bash
uv sync         # install dependencies
uv sync --extra onnx  # optional: fast ONNX Runtime backend (see --backend onnx)
```

**Dependencies:** `torch`, `transformers` (local PII model), `starlette` + `uvicorn` (proxy server), `httpx` (upstream client), `anthropic` + `prompt-toolkit` (demo scripts).

---

## Running the proxy server

```bash
uv run python -m anon_proxy.server [options]
```

| Flag | Default | Purpose |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address (`0.0.0.0` to expose on LAN) |
| `--port` | `8080` | Listen port |
| `--backend` | `auto` | PII detection backend. `auto` uses a CUDA GPU if present, else CPU. `cuda`/`cpu`/`mps` pin the torch device (`mps` is not auto-picked — it is slower than CPU for this model). `onnx` runs the pre-quantized q4f16 export via ONNX Runtime — much faster on CPU; needs `uv sync --extra onnx`. See [Fast ONNX backend](#fast-onnx-backend). |
| `--extra-upstream` | — | Add custom provider: `name=url[;adapter=anthropic\|openai][;path_prefix=/path]` |
| `--store <file>` | — | Path to persistent PII mapping store. Loaded at startup; saved after each request with new entries. Enables cross-restart placeholder consistency — see [Persistent store](#persistent-store) below. |
| `--multi-user` | off | Namespace PII stores by client credential. Requires each masking request to include `x-api-key` or `authorization`; with `--store`, the path is treated as a directory. |
| `--debug` | off | Log new store entries and masked/unmasked diffs to stderr |
| `--config <file>` | — | Unified `config.json` (extra regex patterns, per-label merge-gap overrides, ML labels to skip masking on). See [Config file](#config-file) below. |
| `--no-default-patterns` | off | Disable built-in regex detectors for common PII and secrets |
| `--canary warn\|fix\|off` | `warn` | Run regex detectors after masking; `fix` masks any surviving hit before forwarding |
| `--min-known-entity-len <N>` | `6` | Minimum stored value length for exact known-entity matching; `0` disables |
| `--chunk-size <N>` | `6000` | Max chars per model inference chunk — lower values reduce peak VRAM |
| `--batch-size <N>` | `8` | Batch size for model inference over chunks |
| `--no-system-inject` | off | Disable the placeholder-explainer system prompt that the proxy prepends to outbound requests. Also settable via `system_inject: false` in `config.json`. |

**Add a custom provider:**
```bash
uv run python -m anon_proxy.server \
  --extra-upstream "myprovider=https://api.example.com;adapter=anthropic"
```

Then use: `base_url=http://127.0.0.1:8080/myprovider`

**With config file:**
```bash
uv run python -m anon_proxy.server \
  --config config.json \
  --backend mps \
  --debug
```

### Fast ONNX backend

`--backend onnx` runs the pre-quantized `model_q4f16.onnx` export that the
`openai/privacy-filter` repo already ships, through ONNX Runtime, instead of
the default torch pipeline. On CPU-only machines (the common laptop and k8s
case) this is dramatically faster with no loss of detection coverage.

```bash
uv sync --extra onnx
uv run python -m anon_proxy.server --backend onnx
```

- **Opt-in:** the onnx extra (`onnxruntime`) is not part of the base
  install. Without it, `--backend onnx` fails with a clear install hint.
- **Same detections:** a golden parity gate
  (`tests/test_backend_parity.py`) asserts the onnx backend masks every
  character the torch backend masks across names, emails, phones, addresses,
  a chunk-boundary case, CJK, and code — over-masking is allowed, missing a
  torch detection fails the gate.
- **First run downloads** the ~0.77 GB q4f16 graph (plus its weights sidecar)
  from Hugging Face; subsequent runs are cached.

**Benchmark** — 12-turn agent replay, each request carrying the full history
(the dominant real shape), on a CPU-only Apple Silicon laptop:

| backend | cold start | warm median | warm p95 | 12-turn total |
|---------|-----------:|------------:|---------:|--------------:|
| torch (default) | 7.5 s | 17.5 s | 44.8 s | 230.1 s |
| **onnx q4f16** | 7.9 s | **1.9 s** | 3.2 s | **27.5 s** |

≈9× faster on warm turns and ≈8× lower total latency, with no missed
detections (the parity gate above). The win comes from int4/fp16 quantization,
so it is largest exactly where torch hurts most: CPU-only boxes.

Reproduce the benchmark yourself:

```bash
ANON_PROXY_LIVE_TESTS=1 uv run --extra onnx python scripts/bench_masking.py
```

### Config file

`config.json` is a single JSON object with optional top-level keys:

```json
{
  "patterns": {
    "SSN":  "\\b\\d{3}-\\d{2}-\\d{4}\\b",
    "IPV4": "\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b"
  },
  "merge_gap": {
    "PERSON": " \t\n-'.",
    "EMAIL":  ""
  },
  "ignore_labels": ["DATE", "TITLE"],
  "system_inject": true,
  "default_patterns": true,
  "canary": "warn",
  "min_known_entity_len": 6,
  "upstreams": {
    "deepseek": {
      "base_url": "https://api.deepseek.com",
      "adapter": "anthropic",
      "path_prefix": "anthropic"
    }
  }
}
```

- **`patterns`** — extra regex detectors for PII the ML model misses (SSNs, IPs, internal IDs). Run *before* the ML pass; matches are substituted inline so the model still sees full sentence context. Same-label entries override built-in default patterns.
- **`merge_gap`** — per-label characters allowed inside a gap when merging adjacent same-label spans (e.g. hyphen for `PERSON` so "Jean-Luc" → one token). Overrides entries in the model's defaults; labels you don't mention keep the default.
- **`ignore_labels`** — labels detected by the ML model that should *not* be masked. Useful for noisy categories (e.g. `DATE`, `TITLE`) that confuse the upstream LLM more than they protect privacy. Regex matches are unaffected — if you don't want a regex label, just don't include it in `patterns`.
- **`system_inject`** *(default `true`)* — prepend a short system prompt to outbound requests telling the model that `<LABEL_N>` tokens are opaque references it should echo verbatim, not invent fill-in values for. Merged with any system prompt the client already sent (so client `cache_control` markers on later blocks stay valid). Disable if you've already embedded equivalent instructions client-side, or pass `--no-system-inject` on the command line.
- **`default_patterns`** *(default `true`)* — enable built-in regex detectors for common emails, phone numbers, SSNs, IPv4 addresses, credit cards with separators, and high-structure secrets such as AWS keys, GitHub tokens, JWTs, private-key headers, and Slack tokens. Disable with `false` or `--no-default-patterns`.
- **`canary`** *(default `"warn"`)* — run regex detectors over the final masked text. `"warn"` logs any surviving hit; `"fix"` masks it before forwarding; `"off"` disables the check.
- **`min_known_entity_len`** *(default `6`)* — once a value is learned, exact later occurrences of at least this length are masked anywhere, including code, logs, and JSON. Set `0` to disable exact known-entity matching.
- **`upstreams`** — extra upstream providers, keyed by URL-prefix name. Each entry needs `base_url`; `adapter` (`"anthropic"` or `"openai"`, default `"anthropic"`), `path_prefix`, and `sse` are optional. Same shape as `--extra-upstream`; CLI flags override config entries with the same name.

See [`config.json`](config.json) at the repo root for a working example.

### Persistent store

By default, placeholder mappings live only in memory and are lost when the proxy restarts. Pass `--store` to persist them to disk:

```bash
uv run python -m anon_proxy.server --store /data/pii_store.json
```

The store file is loaded at startup and updated after each request that discovers new PII. On restart the proxy picks up where it left off, so `<PERSON_1>` still refers to the same person.

The store is a flat JSON file — human-readable, easy to inspect, backup, or pre-seed:

```json
{
  "reverse": {
    "<PERSON_1>": "Alice Smith",
    "<EMAIL_1>": "alice@company.com",
    "<PHONE_1>": "555-867-5309"
  },
  "counters": {
    "PERSON": 2,
    "EMAIL": 2,
    "PHONE": 2
  }
}
```

Writes are atomic (written to a `.tmp` file, then renamed) and offloaded to a thread pool so they never block the event loop. If a write fails (disk full, permissions), the error is logged to stderr and the request completes normally — the mapping survives in memory and will be retried on the next write.

Also settable via `ANON_PROXY_STORE` environment variable.

### Multi-user deployments

Single-user mode uses one shared placeholder store for the whole proxy process.
That is right for a local developer proxy, but unsafe for a shared deployment:
any client that can send `<PERSON_1>` could otherwise learn another client's
mapping if the response is unmasked through the same store.

Use `--multi-user` when multiple clients share one proxy:

```bash
uv run python -m anon_proxy.server --multi-user --store /data/pii-stores
```

In multi-user mode, anon-proxy derives a client namespace from the upstream
credential in `x-api-key` or `authorization`. Requests that need masking and
do not include either header fail closed with `401`.

With `--store`, the path is a directory. Each client gets its own
`<client_id>.json` file, where `client_id` is the first 16 hex characters of a
SHA-256 hash of the credential. The credential itself is never written to disk.

### Managing the placeholder store

Inspect or clean a store with `anon-proxy-store` after stopping the proxy:

```bash
uv run anon-proxy-store --store /data/pii_store.json list --label PERSON
uv run anon-proxy-store --store /data/pii_store.json show '<PERSON_1>'
uv run anon-proxy-store --store /data/pii_store.json purge '<PERSON_42>'
```

Bulk-prune short false-positive fragments with a dry run first:

```bash
uv run anon-proxy-store --store /data/pii_store.json prune --label PERSON --max-len 3 --dry-run
uv run anon-proxy-store --store /data/pii_store.json prune --label PERSON --max-len 3
```

`purge` and non-dry-run `prune` write `/data/pii_store.json.bak` before
modifying the store. Counters are never decremented, so deleted placeholder
indexes are not reused in old transcripts. `prune` requires at least one filter
or an explicit `--all`; this prevents an accidental bare prune from selecting
the entire store.

## Menu-bar indicator (macOS)

A menu-bar dinosaur whose run speed tracks live `/_status` token throughput,
with idle, masking-error alarm, down states, per-agent attribution, holiday
skins, and a terminal fallback for non-macOS hosts.

```bash
uv sync --extra menubar

# With the proxy already running:
uv run anon-proxy-menubar
uv run anon-proxy-menubar --watch
uv run anon-proxy-menubar --url http://127.0.0.1:8080/_status
```

The dropdown includes:

- `Theme`: `Auto`, `Classic`, `Halloween`, and `Winter`. `Auto` switches by date.
- `Reset alarm`: re-arms the masking-error latch after you have inspected it.
- `Start proxy`, `Stop proxy`, `Restart proxy`: supervises only a proxy process
  launched by this menu-bar app.
- `Start at login`: installs or removes the launchd agent
  `com.anon-proxy.menubar`.

Regenerate the committed dino frames after editing the pixel matrices:

```bash
uv run --extra gen python scripts/gen_dino_assets.py
```

Thin-shell macOS smoke test:

```bash
uv sync --extra menubar
uv run python -m anon_proxy.server --port 8080
uv run anon-proxy-menubar --url http://127.0.0.1:8080/_status
```

Verify by eye that the dinosaur appears in the menu bar, stands still while
idle, switches to the running frames when `tokens_per_sec` rises, shows the last
client in the dropdown, changes theme from the menu, and falls back to the down
state when the proxy exits.

## Docker

A CPU-only image is provided (~330MB on x86_64, ~1.4GB on aarch64 — PyTorch's ARM wheel is chunkier). Model weights are **not** baked in; they're downloaded into `/models` on first run, so persist that volume to avoid re-downloading on every restart.

```bash
docker build -t anon-proxy:latest .
docker run --rm -p 8080:8080 -v anon-proxy-models:/models anon-proxy:latest
```

**Mount points:**

| Path | Purpose |
|---|---|
| `/config` | Read-only. Drop in `config.json`; the entrypoint auto-discovers it. |
| `/models` | Read-write. `HF_HOME` — privacy-filter weights live here. Persist this. |
| `/data`   | Read-write. Destination for `capture.jsonl`, `pii_store.json`, and other runtime output. |

**Configuration:** every CLI flag also reads from an `ANON_PROXY_*` env var (`-e ANON_PROXY_DEBUG=true`, `-e ANON_PROXY_BACKEND=cpu`, etc.). Any extra args after the image name are forwarded to the server.

**Kubernetes:** see the header comments in [`Dockerfile`](Dockerfile) for a pod-spec sketch with ConfigMap/PVC mounts. If you create a Service named `anon-proxy`, set `enableServiceLinks: false` on the pod — otherwise k8s injects `ANON_PROXY_PORT=tcp://...` and clobbers the app's own port env var.

---

## Testing with the proxy

Test the PII masking through the proxy using `test_mask.py`:

```bash
# Start the proxy
uv run python -m anon_proxy.server --debug

# In another terminal, test with Anthropic (--no-mask means proxy handles masking)
ANTHROPIC_API_KEY=sk-ant-... \
ANTHROPIC_BASE_URL=http://127.0.0.1:8080/anthropic \
uv run python test_mask.py --provider anthropic --no-mask

# Or test with OpenAI
OPENAI_API_KEY=sk-... \
OPENAI_BASE_URL=http://127.0.0.1:8080/openai \
uv run python test_mask.py --provider openai --no-mask
```

---

## Using with Claude Code

Point Claude Code at the proxy (note the provider prefix in the URL):

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080/anthropic claude
```

Or set it permanently in `~/.zshrc` / `~/.bashrc`:
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080/anthropic
```

No other changes — the proxy forwards your auth headers unchanged.

## Using with OpenAI SDK

For OpenAI-compatible clients, use the `/openai` provider path:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8080/openai python your_openai_app.py
```

Or export it permanently:
```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/openai
```

## Debug output

With `--debug`, each request prints a compact diff to stderr:
```
==== anthropic /v1/messages | model=claude-opus-4-7 | 3 msg ====
[store +2]
  <PERSON_1>  ←  'Alice Smith'
  <EMAIL_1>   ←  'alice@company.com'
[masked]
  user[2] text: 'Fix the bug reported by Alice Smith (alice@company.com)…'
              → 'Fix the bug reported by <PERSON_1> (<EMAIL_1>)…'
[unmasked stream] 'I'll fix the bug for <PERSON_1>…' → 'I'll fix the bug for Alice Smith…'
```

**What gets protected:** every user and assistant message turn — text content, tool call inputs (`tool_use.input`), tool results (`tool_result.content`), and Claude Code system-reminder blocks. File contents, shell output, names, emails, paths containing PII are all masked before leaving your machine.

**What is NOT masked:** the system prompt (tool schemas and static instructions), tool definitions, and extended-thinking blocks (signatures would break). See [`SECURITY.md`](SECURITY.md) for the full threat model and known limitations.

**How it works:** PII spans get stable placeholder tokens (`<PERSON_1>`, `<EMAIL_1>`, `<ADDRESS_1>`, …) stored in an in-memory mapping. The same value always maps to the same token across turns so the model stays coherent. Once a value is learned, exact later occurrences are masked anywhere, including code, logs, and JSON; mask-cache entries computed before a value was learned are not retroactively rewritten. Optionally persist this mapping to disk with `--store` (see [Persistent store](#persistent-store)). Responses are unmasked before reaching your client.

---

## FAQ

### How is this different from Microsoft Presidio?

Presidio is a Python library for PII detection — you call it from your code. anon-proxy is a transparent network proxy: you point your existing LLM client at it via `ANTHROPIC_BASE_URL` or `OPENAI_BASE_URL` and it intercepts every request. It also maintains a stable token↔value mapping across turns, so the model sees the same `<PERSON_1>` on turn 5 that it saw on turn 1 (Presidio doesn't do this — that's an application-layer concern it leaves to you).

### Does this work with Claude Code?

Yes. That's the primary supported client. Set `ANTHROPIC_BASE_URL=http://127.0.0.1:8080/anthropic` and run `claude` normally. See [Using with Claude Code](#using-with-claude-code).

### Does this work with OpenAI / ChatGPT SDK clients?

Yes — point your OpenAI client at `http://127.0.0.1:8080/openai`. See [Using with OpenAI SDK](#using-with-openai-sdk).

### Will the LLM still understand my prompt after PII is replaced with placeholders?

Generally yes. Modern LLMs treat `<PERSON_1>` and `<EMAIL_1>` as opaque variable names and reason about relationships between them. The stable mapping (same person → same token across turns) is what makes multi-turn conversations work — without it, "tell <PERSON_1> about the meeting" on turn 2 would refer to a different person than turn 1's `<PERSON_1>`.

### Does masking break tool calls?

Tool inputs and tool results are masked/unmasked just like message text, so most tools work unchanged. Caveat: if a tool's behavior depends on the literal value of a PII string (e.g. a database lookup that takes an email), you'll want to register that field in the config's `patterns` carefully — or skip masking for that tool. Extended-thinking blocks are passed through unmasked because their cryptographic signatures would break.

### What PII does it detect?

Out of the box: persons, emails, phone numbers, addresses, organizations, dates of birth, government IDs, common clue-less regex shapes, secrets, and other categories from the openai/privacy-filter model. Add your own (internal employee IDs, project codenames) via the config's `patterns` section.

### What's the performance overhead?

We don't have published benchmark numbers yet — latency measurement is on the roadmap.

### Is the threat model documented?

Yes — see [`SECURITY.md`](SECURITY.md). It covers what's in scope (request bodies leaving your machine through the supported adapters), what's out of scope (a malicious local user, side channels, the system prompt itself), and known false-negative modes.

### How do I report a vulnerability?

See [`SECURITY.md`](SECURITY.md). For a privacy tool, *quietly* is usually better than a public issue — please email the maintainer first.

---

## Next steps / roadmap

- **Quality assurance** : Enhance PII detection quality tracking and add comprehensive unit/integration tests with benchmarking.
- **Observability** : Implement structured logging and telemetry for monitoring proxy performance and PII masking metrics.
- ~~**Persistence** : PII mappings can be persisted to disk via `--store` so placeholder consistency survives server restarts.~~ ✅
- **Usability** : Now supporting Anthropic and OpenAI APIs, but need more compatibility testing and expand to other potential providers.
- **Dev infrastructure** : Set up CI, contribution guidelines, and project templates to streamline community development.

---

## License & security

- Licensed under the [MIT License](LICENSE).
- For security disclosures and the threat model, see [`SECURITY.md`](SECURITY.md).
- Issues and PRs welcome — this is a young project and feedback is the fastest way to improve it.
