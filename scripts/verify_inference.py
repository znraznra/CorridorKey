"""Run the official trained model on synthetic plates and record real GPU evidence.

This is a numerical/memory smoke test, not a real-footage quality benchmark.
"""

# Environment variables must be set before importing OpenCV and the engine.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CORRIDORKEY_SKIP_COMPILE"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch

from CorridorKeyModule.backend import create_engine
from device_utils import resolve_device
from scripts.gpu_check import inspect_gpu


def synthetic_plate(phase=0):
    """Known foreground with soft edges, narrow lines and translucent detail."""
    h, w = 720, 1280
    alpha = np.zeros((h, w), np.float32)
    shift = phase * 5
    cv2.ellipse(alpha, (w // 2 + shift, 370), (170, 245), 0, 0, 360, 1.0, -1)
    cv2.circle(alpha, (w // 2 + shift, 175), 92, 1.0, -1)
    for x in range(420, 860, 17):
        cv2.line(alpha, (x + shift, 360), (x - 70 + shift, 150), 0.6, 2)
    cv2.rectangle(alpha, (810 + shift, 310), (1010 + shift, 570), 0.35, -1)
    alpha = cv2.GaussianBlur(alpha, (7, 7), 1.2)
    y, x = np.mgrid[:h, :w]
    fg = np.stack((0.65 + 0.13 * np.sin(x / 34), 0.18 + 0.06 * np.cos(y / 28), 0.15 + 0.09 * x / w), axis=-1)
    bg = np.zeros_like(fg)
    bg[:] = (0.06, 0.78, 0.08)
    rgb = fg * alpha[..., None] + bg * (1 - alpha[..., None])
    hint = cv2.erode((alpha > 0.65).astype(np.float32), np.ones((17, 17), np.uint8))
    hint = cv2.GaussianBlur(hint, (31, 31), 6)
    return np.round(rgb * 255).astype(np.uint8), hint, alpha


def write_rgb(path, value):
    assert cv2.imwrite(str(path), cv2.cvtColor(value, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--image-size", type=int, choices=(512, 1024, 2048), default=512)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--compare-cpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    logging.basicConfig(level=logging.INFO)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(resolve_device(args.device))
    backend = getattr(torch, device.type)
    report = {"saved_at_utc": datetime.now(timezone.utc).isoformat(), "hardware": inspect_gpu(args.device)}
    report["fixture"] = "synthetic 1280x720 plate; official trained weights; no real-footage quality claim"
    engine = create_engine(backend="torch", device=str(device), img_size=args.image_size)
    report["checkpoint_sha256"] = hashlib.sha256(Path(engine.checkpoint_path).read_bytes()).hexdigest()
    observed_devices = []
    hook = engine.model.register_forward_pre_hook(lambda _, inputs: observed_devices.append(str(inputs[0].device)))
    image, hint, expected = synthetic_plate()
    backend.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    warmup = engine.process_frame(image, hint, post_process_on_gpu=False, auto_despeckle=False)
    backend.synchronize(device)
    report["warmup_seconds"] = time.perf_counter() - begin
    times = []
    for i in range(args.frames):
        image_i, hint_i, _ = synthetic_plate(i)
        begin = time.perf_counter()
        output = engine.process_frame(image_i, hint_i, post_process_on_gpu=False, auto_despeckle=False)
        backend.synchronize(device)
        times.append(time.perf_counter() - begin)
        for key, channels in (("alpha", 1), ("fg", 3), ("processed", 4), ("comp", 3)):
            value = output[key]
            assert value.shape == (720, 1280, channels), (key, value.shape)
            assert np.isfinite(value).all(), key
        alpha_u8 = np.clip(output["alpha"][..., 0] * 255, 0, 255).astype(np.uint8)
        comp_u8 = np.clip(output["comp"] * 255, 0, 255).astype(np.uint8)
        write_rgb(args.output / f"comp_{i:04d}.png", comp_u8)
        if i == 0:
            write_rgb(args.output / "input.png", image)
            cv2.imwrite(str(args.output / "hint.png"), np.round(hint * 255).astype(np.uint8))
            cv2.imwrite(str(args.output / "expected_alpha.png"), np.round(expected * 255).astype(np.uint8))
            cv2.imwrite(str(args.output / "predicted_alpha.png"), alpha_u8)
            assert cv2.imwrite(
                str(args.output / "processed.exr"), cv2.cvtColor(output["processed"], cv2.COLOR_RGBA2BGRA)
            )
            tiles = [
                image,
                np.repeat(np.round(hint[..., None] * 255).astype(np.uint8), 3, axis=-1),
                np.repeat(alpha_u8[..., None], 3, axis=-1),
                comp_u8,
            ]
            panel = np.concatenate([cv2.resize(tile, (640, 360)) for tile in tiles[:2]], axis=1)
            bottom = np.concatenate([cv2.resize(tile, (640, 360)) for tile in tiles[2:]], axis=1)
            panel = np.concatenate((panel, bottom), axis=0)
            for label, xy in (
                ("Synthetic input", (15, 30)),
                ("Coarse hint", (655, 30)),
                ("Trained model alpha", (15, 390)),
                ("Composite", (655, 390)),
            ):
                cv2.putText(panel, label, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            write_rgb(args.output / "comparison.png", panel)
    hook.remove()
    assert all(item == str(next(engine.model.parameters()).device) for item in observed_devices)
    report.update(
        {
            "image_size": args.image_size,
            "model_precision": str(engine.model_precision),
            "compiled": engine._compiled,
            "observed_model_input_devices": observed_devices,
            "seconds_per_frame": times,
            "mean_seconds_per_frame": sum(times) / len(times),
            "peak_allocated_bytes": backend.max_memory_allocated(device),
            "peak_reserved_bytes": backend.max_memory_reserved(device),
            "output_shapes_and_finite_values": "passed",
        }
    )
    if args.compare_cpu:
        # Use the identical weights/model and FP32 reference; retain the GPU
        # model's precision report above before moving it to CPU.
        engine.model = engine.model.to(device="cpu", dtype=torch.float32)
        engine.device = torch.device("cpu")
        engine.mean = engine.mean.cpu().float()
        engine.std = engine.std.cpu().float()
        engine.model_precision = torch.float32
        engine.mixed_precision = False
        reference = engine.process_frame(image, hint, post_process_on_gpu=False, auto_despeckle=False)
        errors = {key: float(np.abs(warmup[key] - reference[key]).mean()) for key in ("alpha", "fg")}
        report["cpu_reference_mean_absolute_error"] = errors
        assert max(errors.values()) < 0.005, errors
    report["status"] = "passed"
    result = json.dumps(report, indent=2)
    (args.output / "verification.json").write_text(result + "\n", encoding="utf-8")
    print(result, flush=True)


if __name__ == "__main__":
    main()
