"""Exercise the real XPU BF16 operators used by CorridorKey 2, without weights."""

import argparse
import json
import sys
from pathlib import Path


def inspect(device):
    import torch
    import torch.nn.functional as functional
    from corridorkey.accelerator import empty_cache, require_device, synchronize

    device = require_device(device)
    if device.type != "xpu":
        raise ValueError("This experimental integration checks XPU only.")
    torch.xpu.reset_peak_memory_stats(device)
    x = torch.ones((1, 4, 16, 16), device=device)
    weight = torch.ones((8, 4, 3, 3), device=device)
    with torch.inference_mode(), torch.autocast("xpu", dtype=torch.bfloat16):
        conv = functional.conv2d(x, weight)
        matrix = torch.ones((32, 32), device=device) @ torch.ones((32, 32), device=device)
        q = torch.ones((1, 2, 8, 16), device=device, dtype=torch.bfloat16)
        attention = functional.scaled_dot_product_attention(q, q, q)
        norm = functional.rms_norm(q, (16,))
        resized = functional.interpolate(x, (8, 8), mode="area")
    synchronize(device)
    for value in (conv, matrix, attention, norm, resized):
        if value.device != device or not torch.isfinite(value).all().item():
            raise RuntimeError("XPU operation returned a wrong-device or non-finite result.")
    if not torch.all(conv == 36).item() or not torch.all(matrix == 32).item():
        raise RuntimeError("XPU convolution/matmul numerical check failed.")
    if not torch.all(attention == 1).item() or not torch.all(resized == 1).item():
        raise RuntimeError("XPU attention/interpolation numerical check failed.")
    if conv.dtype != torch.bfloat16 or matrix.dtype != torch.bfloat16:
        raise RuntimeError("Expected XPU BF16 autocasting did not occur.")
    report = {
        "engine": "v2", "experimental": True, "torch": torch.__version__,
        "device": str(device), "name": torch.xpu.get_device_name(device),
        "total_memory": torch.xpu.get_device_properties(device).total_memory,
        "bf16_conv_matmul_sdpa_rmsnorm_area": "passed",
        "peak_allocated_bytes": torch.xpu.max_memory_allocated(device),
        "pretrained_inference_validated": False,
        "video_encoding_validated": False,
    }
    empty_cache(device)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    report = json.dumps(inspect(args.device), indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
