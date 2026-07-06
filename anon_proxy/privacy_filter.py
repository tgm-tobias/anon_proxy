from dataclasses import dataclass
from typing import Literal

from transformers import pipeline


@dataclass(frozen=True)
class PIIEntity:
    label: str
    text: str
    start: int
    end: int
    score: float


# Per-label characters allowed inside the gap between two adjacent same-label
# spans when deciding whether to merge them. Labels absent from this map can
# only merge across an empty gap.
DEFAULT_MERGE_GAP_ALLOWED: dict[str, str] = {
    # Names: space for "Alice Smith", hyphen for "Jean-Luc", apostrophe for
    # "O'Neil", period for "J.R.".
    "PERSON": " \t\n-'.",
    # Addresses: whitespace plus the punctuation that commonly splits a
    # single postal address — "123 Main St., Apt #4", "Suite 3B-1".
    "ADDRESS": " \t\n,.#-/",
    # Dates: "Jan 1, 2025", "2025-01-01", "1/1/25".
    "DATE": " \t\n,/.-",
    "ORGANIZATION": " \t\n&.,-",
    "LOCATION": " \t\n,.-",
}

# Backends. "auto"/"torch"/"cpu"/"mps" all run the torch HF pipeline (cpu/mps
# just pin the device); "onnx" loads the pre-quantized graph the model repo
# already ships and runs it through ONNX Runtime. The torch path is unchanged.
Backend = Literal["auto", "torch", "cpu", "mps", "onnx"]

# Upstream ships this pre-quantized export in the model repo; we load it, never
# convert. q4f16 is the smallest (~0.77 GB) and the parity/bench winner.
ONNX_SUBFOLDER = "onnx"
ONNX_FILE = "model_q4f16.onnx"
# The q4f16 export stores its weights in an external data sidecar that ORT
# loads implicitly from the same directory; it must be fetched alongside.
ONNX_DATA_FILE = "model_q4f16.onnx_data"
DEFAULT_ONNX_PROVIDER = "CPUExecutionProvider"


class PrivacyFilter:
    """Thin wrapper around the openai/privacy-filter token classifier.

    The HF pipeline's aggregation_strategy only merges subword pieces within
    a single word. For PII masking we also want to merge *adjacent* same-label
    spans ("Alice" + "Smith" → "Alice Smith", "alice@example" + ".com" →
    "alice@example.com") so each entity maps to one placeholder downstream.
    That second merge pass is `merge_adjacent`, on by default.

    The merge rule is per-label: two same-label spans merge iff either the
    gap between them is empty, or every character of the gap appears in that
    label's allowed-char set. Defaults live in `DEFAULT_MERGE_GAP_ALLOWED`
    (whitespace is NOT automatically mergeable — it has to be in the label's
    entry). Pass `merge_gap_allowed` to override per-label entries; labels
    you don't mention keep their default, and labels you set to `""` can only
    merge across an empty gap.

    Long texts are split into overlapping-free chunks of at most `chunk_size`
    characters (default 1500, ~375 English tokens — safely within BERT's 512
    token limit). Splits happen at the last whitespace before the boundary so
    words are never bisected. Entity spans from adjacent chunks are combined
    before the adjacency-merge pass, so entities that straddle a chunk boundary
    are still collapsed into a single placeholder.
    """

    MODEL_ID = "openai/privacy-filter"

    def __init__(
        self,
        *,
        aggregation_strategy: str = "simple",
        merge_adjacent: bool = True,
        merge_gap_allowed: dict[str, str] | None = None,
        chunk_size: int = 1500,
        device: int | str | None = None,
        backend: Backend = "auto",
        onnx_provider: str = DEFAULT_ONNX_PROVIDER,
    ) -> None:
        self._pipe = _build_pipeline(
            model_id=self.MODEL_ID,
            aggregation_strategy=aggregation_strategy,
            backend=backend,
            device=device,
            onnx_provider=onnx_provider,
        )
        self._merge_adjacent = merge_adjacent
        merged_policy = {**DEFAULT_MERGE_GAP_ALLOWED, **(merge_gap_allowed or {})}
        self._gap_allowed: dict[str, frozenset[str]] = {
            label: frozenset(chars) for label, chars in merged_policy.items()
        }
        self._chunk_size = chunk_size

    def detect(self, text: str) -> list[PIIEntity]:
        if not text.strip():
            return []
        chunks = _split_chunks(text, self._chunk_size)
        entities: list[PIIEntity] = []
        for offset, chunk in chunks:
            for r in self._pipe(chunk):
                e = _to_entity(r, chunk)
                if e is None:
                    continue
                entities.append(
                    PIIEntity(
                        label=e.label,
                        text=e.text,
                        start=e.start + offset,
                        end=e.end + offset,
                        score=e.score,
                    )
                )
        if self._merge_adjacent:
            entities = _merge_adjacent_entities(entities, text, self._gap_allowed)
        return entities

    def detect_raw(self, text: str) -> list[dict]:
        """Return the pipeline's untouched per-span dicts for debugging."""
        return list(self._pipe(text))


def _build_pipeline(
    *,
    model_id: str,
    aggregation_strategy: str,
    backend: Backend,
    device: int | str | None,
    onnx_provider: str,
):
    """Construct the callable that `detect()` invokes once per chunk.

    torch/auto/cpu/mps → the HF token-classification pipeline (unchanged).
    onnx → the ONNX Runtime classifier, exposing the same call surface.
    """
    if backend in ("auto", "torch", "cpu", "mps"):
        if backend in ("cpu", "mps") and device is None:
            device = backend
        return pipeline(
            task="token-classification",
            model=model_id,
            aggregation_strategy=aggregation_strategy,
            device=device,
        )
    if backend == "onnx":
        if device is not None:
            raise ValueError("device is only valid with a torch backend")
        return _load_onnx_classifier(model_id=model_id, provider=onnx_provider)
    raise ValueError(
        f"unsupported backend {backend!r}; expected one of "
        "'auto', 'torch', 'cpu', 'mps', 'onnx'"
    )


def _load_onnx_classifier(*, model_id: str, provider: str):
    """Load the q4f16 ONNX export and wrap it in the pipeline call surface.

    We call ONNX Runtime directly rather than via optimum: optimum caps
    transformers below 4.58, incompatible with this project's transformers 5.
    """
    import json

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "the onnx backend requires the optional onnx dependencies; "
            "install them with `uv sync --extra onnx`"
        ) from e
    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover - transformers is a base dep
        raise RuntimeError(
            "transformers and huggingface_hub are required to load the ONNX model"
        ) from e

    model_path = hf_hub_download(model_id, filename=f"{ONNX_SUBFOLDER}/{ONNX_FILE}")
    # The external-weights sidecar is loaded implicitly by ORT from the same
    # directory; fetch it so it is on disk next to model_q4f16.onnx.
    hf_hub_download(model_id, filename=f"{ONNX_SUBFOLDER}/{ONNX_DATA_FILE}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Read id2label straight from config.json — no AutoConfig, no
    # trust_remote_code, no custom-architecture resolution needed.
    config_path = hf_hub_download(model_id, filename="config.json")
    with open(config_path, encoding="utf-8") as f:
        id2label = json.load(f)["id2label"]
    session = ort.InferenceSession(model_path, providers=[provider])
    return _OnnxTokenClassifier(session, tokenizer, id2label)


class _OnnxTokenClassifier:
    """Minimal ONNX Runtime stand-in for the HF pipeline call surface.

    `detect()` calls this once per chunk with a single string; a list is
    accepted too for parity with the pipeline. Returns the fields
    `_to_entity` consumes:
    `entity_group`, `start`, `end`, `word`, `score`. Aggregation matches HF
    'simple' over the model's BIOES tags: argmax per token, strip the B/I/E/S-
    prefix, merge consecutive tokens sharing a base label, span score = min of
    member-token softmax maxima.
    """

    def __init__(self, session, tokenizer, id2label: dict) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._id2label = {int(k): v for k, v in id2label.items()}
        self._input_names = {i.name for i in session.get_inputs()}

    def __call__(self, inputs, **_kwargs):
        if isinstance(inputs, str):
            return self._detect_one(inputs)
        return [self._detect_one(text) for text in inputs]

    def _detect_one(self, text: str) -> list[dict]:
        encoded = self._tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="np",
            truncation=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        feeds = {name: encoded[name] for name in self._input_names if name in encoded}
        logits = self._session.run(None, feeds)[0][0]
        probs = _softmax(logits)
        label_ids = probs.argmax(axis=-1)
        scores = probs.max(axis=-1)

        tokens: list[dict] = []
        for label_id, score, (start, end) in zip(label_ids, scores, offsets):
            if start == end:  # special / padding token
                continue
            label = self._id2label.get(int(label_id), "O")
            if label == "O":
                continue
            prefix, entity = _split_tag(label)
            tokens.append(
                {
                    "prefix": prefix,
                    "entity": entity,
                    "start": int(start),
                    "end": int(end),
                    "score": float(score),
                }
            )
        return _aggregate_tokens(tokens, text)


def _softmax(logits):
    import numpy as np

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _split_tag(label: str) -> tuple[str, str]:
    """Split a BIOES tag like 'B-private_person' into ('B', 'private_person').

    Tags without a recognized single-letter prefix are treated as a bare
    begin ('B', <label>).
    """
    if len(label) > 2 and label[1] == "-" and label[0] in "BIES":
        return label[0], label[2:]
    return "B", label


def _aggregate_tokens(tokens: list[dict], text: str) -> list[dict]:
    spans: list[dict] = []
    current: dict | None = None
    for token in tokens:
        starts_new = (
            current is None
            or token["prefix"] in ("B", "S")
            or token["entity"] != current["entity_group"]
        )
        if starts_new:
            if current is not None:
                spans.append(current)
            current = {
                "entity_group": token["entity"],
                "start": token["start"],
                "end": token["end"],
                "score": token["score"],
            }
        else:
            current["end"] = token["end"]
            current["score"] = min(current["score"], token["score"])
        if token["prefix"] == "S":
            spans.append(current)
            current = None
    if current is not None:
        spans.append(current)
    for span in spans:
        span["word"] = text[span["start"] : span["end"]]
    return spans


def _split_chunks(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Return (start_offset, chunk_text) pairs covering all of `text`.

    Each chunk is at most `max_chars` characters. Splits at the last
    whitespace before the limit so words are not bisected; falls back to a
    hard cut if no whitespace is found in the window.
    """
    if len(text) <= max_chars:
        return [(0, text)]
    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        if start + max_chars >= len(text):
            chunks.append((start, text[start:]))
            break
        split = text.rfind(" ", start, start + max_chars)
        if split <= start:
            split = start + max_chars  # hard cut — no whitespace in window
        else:
            split += 1  # include the space in this chunk
        chunks.append((start, text[start:split]))
        start = split
    return chunks


def _to_entity(raw: dict, original: str) -> PIIEntity | None:
    start, end = _tighten(int(raw["start"]), int(raw["end"]), original)
    if start == end:
        return None
    label = raw.get("entity_group") or raw["entity"]
    return PIIEntity(
        label=label,
        text=original[start:end],
        start=start,
        end=end,
        score=float(raw["score"]),
    )


def _merge_adjacent_entities(
    entities: list[PIIEntity],
    original: str,
    gap_allowed: dict[str, frozenset[str]] | None = None,
) -> list[PIIEntity]:
    if not entities:
        return entities
    ordered = sorted(entities, key=lambda e: e.start)
    merged: list[PIIEntity] = []
    for e in ordered:
        if merged:
            prev = merged[-1]
            gap = original[prev.end : e.start]
            if prev.label == e.label and _gap_mergeable(prev.label, gap, gap_allowed):
                start, end = _tighten(prev.start, e.end, original)
                merged[-1] = PIIEntity(
                    label=prev.label,
                    text=original[start:end],
                    start=start,
                    end=end,
                    score=min(prev.score, e.score),
                )
                continue
        merged.append(e)
    return merged


def _gap_mergeable(
    label: str,
    gap: str,
    gap_allowed: dict[str, frozenset[str]] | None,
) -> bool:
    """Empty gap always merges. Otherwise every character of the gap must be
    in `gap_allowed[label]` — whitespace is not a built-in pass; it only
    counts if the label's allowed set includes it.
    """
    if gap == "":
        return True
    allowed = (gap_allowed or {}).get(label)
    if not allowed:
        return False
    return all(c in allowed for c in gap)


def _tighten(start: int, end: int, original: str) -> tuple[int, int]:
    """Shrink [start, end) to exclude leading/trailing whitespace.

    Keeps the invariant `entity.text == original[entity.start:entity.end]`,
    which the mask/unmask layer relies on for offset-correct replacement.
    """
    while start < end and original[start].isspace():
        start += 1
    while end > start and original[end - 1].isspace():
        end -= 1
    return start, end
