"""Backend selection and compatibility regressions; hardware tests run separately."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from device_utils import clear_device_cache, detect_best_device, recommended_model_precision, resolve_device


@pytest.fixture
def intel_gpu(monkeypatch):
    xpu = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        empty_cache=MagicMock(),
        get_device_name=lambda _: "Intel Arc B580",
        get_device_properties=lambda _: SimpleNamespace(total_memory=12 * 1024**3),
        mem_get_info=lambda _: (10 * 1024**3, 12 * 1024**3),
    )
    monkeypatch.setattr(torch, "xpu", xpu)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("CORRIDORKEY_DEVICE", raising=False)
    return xpu


def test_arc_selected_and_explicit_index_validated(intel_gpu):
    assert detect_best_device() == "xpu"
    assert resolve_device("xpu:0") == "xpu:0"
    with pytest.raises(RuntimeError, match="out of range"):
        resolve_device("xpu:1")


def test_explicit_arc_never_silently_falls_back(intel_gpu):
    intel_gpu.is_available = lambda: False
    with pytest.raises(RuntimeError, match="XPU requested"):
        resolve_device("xpu")


def test_arc_cache_clear_uses_intel_backend(intel_gpu):
    clear_device_cache("xpu:0")
    intel_gpu.empty_cache.assert_called_once()


def test_arc_enumeration(intel_gpu, monkeypatch):
    from device_utils import enumerate_gpus

    monkeypatch.setattr("device_utils._enumerate_nvidia", lambda: None)
    monkeypatch.setattr("device_utils._enumerate_amd", lambda: None)
    (gpu,) = enumerate_gpus()
    assert gpu.name == "Intel Arc B580"
    assert gpu.vram_total_gb == 12
    assert gpu.vram_free_gb == 10


@pytest.mark.parametrize(
    "device,capability,expected",
    [
        ("cpu", None, torch.float32),
        ("cuda", (6, 1), torch.float32),
        ("cuda:0", (8, 6), torch.float16),
        ("xpu:0", None, torch.float16),
    ],
)
def test_precision_policy(device, capability, expected, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    assert recommended_model_precision(device) == expected


@pytest.mark.parametrize("device", ["xpu:-1", "xpu:", "cuda:no", "cpu:0"])
def test_invalid_indices_fail_before_inference(device):
    with pytest.raises(RuntimeError):
        resolve_device(device)


def test_legacy_cuda_skips_compiler(monkeypatch):
    from CorridorKeyModule.inference_engine import CorridorKeyEngine

    # Keep tensors on CPU while exercising the real constructor's CUDA policy.
    original_tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: original_tensor(*a, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (6, 1))
    monkeypatch.setattr(CorridorKeyEngine, "_load_model", lambda _: object())
    compile_model = MagicMock()
    monkeypatch.setattr(CorridorKeyEngine, "_compile", compile_model)
    engine = CorridorKeyEngine("not-read", device="cuda", mixed_precision=False)
    compile_model.assert_not_called()
    assert not engine._compiled


def test_xpu_skips_compiler(monkeypatch):
    from CorridorKeyModule.inference_engine import CorridorKeyEngine

    original_tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: original_tensor(*a, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(CorridorKeyEngine, "_load_model", lambda _: object())
    compile_model = MagicMock()
    monkeypatch.setattr(CorridorKeyEngine, "_compile", compile_model)
    CorridorKeyEngine("not-read", device="xpu")
    compile_model.assert_not_called()
