"""Explicit accelerator operations for the experimental CorridorKey 2 port.

Copyright (c) Corridor Digital LLC and contributors. Distributed under the
Corridor Key Licence v2; see the pinned upstream LICENSE fetched by setup.
"""

import torch


def require_device(requested):
    device = torch.device(requested)
    if device.type not in ("cuda", "xpu"):
        raise ValueError("CorridorKey 2 requires an explicit CUDA or Intel XPU device; no CPU fallback.")
    backend = getattr(torch, device.type, None)
    if backend is None or not backend.is_available():
        raise RuntimeError(f"Requested {device.type} GPU is unavailable; no fallback selected.")
    index = device.index if device.index is not None else backend.current_device()
    if index < 0 or index >= backend.device_count():
        raise ValueError(f"GPU index out of range: {requested}")
    device = torch.device(device.type, index)
    backend.set_device(device)
    if device.type == "cuda" and not backend.is_bf16_supported():
        raise RuntimeError("CorridorKey 2 needs BF16 support. Use the v1 p600 profile on Pascal.")
    return device


def runtime_device(model):
    # All model parameters may temporarily be on CPU between staged passes.
    saved = getattr(model, "_corridorkey_device", None)
    return torch.device(saved) if saved is not None else next(model.stage1.parameters()).device


def empty_cache(device):
    device = torch.device(device)
    backend = getattr(torch, device.type)
    with backend.device(device):
        backend.empty_cache()


def synchronize(device):
    device = torch.device(device)
    getattr(torch, device.type).synchronize(device)
