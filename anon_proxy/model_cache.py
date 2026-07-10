"""Explicit, idempotent model weight pre-fetching."""

from __future__ import annotations

from huggingface_hub import snapshot_download

from anon_proxy.privacy_filter import (
    ONNX_DATA_FILE,
    ONNX_FILE,
    ONNX_SUBFOLDER,
    PrivacyFilter,
)

MODEL_ID: str = PrivacyFilter.MODEL_ID

_COMMON_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "*.txt",
]
_TORCH_BACKENDS = {"auto", "torch", "cpu", "mps", "cuda"}
_ONNX_BACKENDS = {"onnx", "onnx-q4f16"}


def normalize_backend(backend: str) -> str:
    """Return the runtime backend name for an onboarding/backend alias."""
    if backend in _TORCH_BACKENDS:
        return backend
    if backend in _ONNX_BACKENDS:
        return "onnx"
    raise ValueError(f"unknown backend {backend!r}")


def files_for_backend(backend: str) -> list[str] | None:
    """Return Hugging Face allow-patterns for a backend, or None for all files."""
    normalized = normalize_backend(backend)
    if normalized in _TORCH_BACKENDS:
        return None
    if normalized == "onnx":
        return _COMMON_FILES + [
            f"{ONNX_SUBFOLDER}/{ONNX_FILE}",
            f"{ONNX_SUBFOLDER}/{ONNX_DATA_FILE}",
        ]
    raise ValueError(f"unknown backend {backend!r}")


def is_cached(backend: str) -> bool:
    """True when the backend's required files already exist locally."""
    try:
        snapshot_download(
            MODEL_ID,
            allow_patterns=files_for_backend(backend),
            local_files_only=True,
        )
    except Exception:
        return False
    return True


def download_model(backend: str = "torch", *, progress: bool = True) -> str:
    """Fetch backend weights into the Hugging Face cache and return the snapshot dir."""
    return snapshot_download(
        MODEL_ID,
        allow_patterns=files_for_backend(backend),
        tqdm_class=None if progress else _SilentTqdm,
    )


class _SilentTqdm:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False
