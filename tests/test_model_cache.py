import pytest

from anon_proxy import model_cache


def test_files_for_torch_is_all():
    assert model_cache.files_for_backend("torch") is None


def test_files_for_onnx_q4f16_is_scoped():
    patterns = model_cache.files_for_backend("onnx-q4f16")
    assert patterns is not None
    assert any("model_q4f16.onnx" in pattern for pattern in patterns)
    assert any("model_q4f16.onnx_data" in pattern for pattern in patterns)
    assert not any("safetensors" in pattern for pattern in patterns)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        model_cache.files_for_backend("mlx")


def test_download_delegates_to_hub(monkeypatch):
    calls = {}

    def fake_snapshot(repo_id, allow_patterns=None, **kwargs):
        calls["repo_id"] = repo_id
        calls["allow_patterns"] = allow_patterns
        calls["kwargs"] = kwargs
        return "/fake/snapshot/dir"

    monkeypatch.setattr(model_cache, "snapshot_download", fake_snapshot)

    out = model_cache.download_model("onnx-q4f16", progress=False)

    assert out == "/fake/snapshot/dir"
    assert calls["repo_id"] == model_cache.MODEL_ID
    assert any("model_q4f16.onnx" in pattern for pattern in calls["allow_patterns"])
