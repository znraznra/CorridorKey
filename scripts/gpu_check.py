"""Verify the selected compute device with real tensor operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from device_utils import recommended_model_precision, resolve_device


def inspect_gpu(requested):
    device = torch.device(resolve_device(requested))
    if device.type not in ("cuda", "xpu", "cpu"):
        raise RuntimeError("This check supports CUDA, Intel XPU or explicit CPU execution.")
    backend = getattr(torch, device.type) if device.type != "cpu" else None
    precision = recommended_model_precision(device)
    if backend:
        backend.reset_peak_memory_stats(device)
    value = torch.ones((1, 4, 64, 64), dtype=precision, device=device)
    weight = torch.ones((8, 4, 3, 3), dtype=precision, device=device)
    conv = torch.nn.functional.conv2d(value, weight)
    matrix = torch.ones((64, 64), dtype=precision, device=device)
    product = matrix @ matrix
    if backend:
        backend.synchronize(device)
    assert conv.device == value.device and product.device == value.device
    assert torch.all(conv == 36).item() and torch.all(product == 64).item()
    result = {
        "torch": torch.__version__,
        "device": str(value.device),
        "name": backend.get_device_name(device) if backend else "CPU",
        "vram_bytes": backend.get_device_properties(device).total_memory if backend else None,
        "precision": str(precision),
        "cuda_runtime": torch.version.cuda,
        "gpu_convolution_and_matmul" if backend else "cpu_convolution_and_matmul": "passed",
    }
    if device.type == "cuda":
        result["compute_capability"] = backend.get_device_capability(device)
        result["compiled_architectures"] = backend.get_arch_list()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(inspect_gpu(args.device), indent=2)
    print(report, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
