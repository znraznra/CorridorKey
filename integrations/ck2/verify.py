"""Run the pretrained CK2 XPU pipeline on a small moving synthetic screen plate."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def inspect_outputs(output, frames, height, width):
    import cv2
    import numpy as np

    report = {}
    for kind, channels in (("alpha", 1), ("fg", 3), ("cutout", 4)):
        directories = list(Path(output).glob(f"*_{kind}_exr"))
        if len(directories) != 1:
            raise RuntimeError(f"Expected one {kind} EXR directory, found {len(directories)}")
        files = sorted(directories[0].glob("*.exr"))
        if len(files) != frames:
            raise RuntimeError(f"{kind}: expected {frames} frames, found {len(files)}")
        minimum, maximum = float("inf"), float("-inf")
        for file in files:
            value = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
            expected = (height, width) if channels == 1 else (height, width, channels)
            if value is None or value.shape != expected or not np.isfinite(value).all():
                raise RuntimeError(f"Invalid EXR dimensions, channels or finite values: {file}")
            minimum, maximum = min(minimum, float(value.min())), max(maximum, float(value.max()))
            if kind == "alpha" and (value.min() < -0.001 or value.max() > 1.001):
                raise RuntimeError(f"Alpha outside [0,1]: {file}")
        if kind == "alpha" and maximum - minimum < 0.01:
            raise RuntimeError("Synthetic alpha is effectively constant; inspect inference.log.")
        report[kind] = {"frames": len(files), "shape": list(expected), "min": minimum, "max": maximum}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh local directory; existing files are preserved")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from check_xpu import inspect

    compute = inspect(args.device)
    import cv2
    import numpy as np

    job = args.output.resolve()
    if job.exists() and any(job.iterdir()):
        raise RuntimeError("Choose a fresh verification output folder; existing files will not be overwritten.")
    plate = job / "plate"
    plate.mkdir(parents=True, exist_ok=True)
    frames, height, width = 6, 256, 384
    for index in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (25, 180, 25)  # OpenCV BGR.
        center = (180 + index * 3, 130)
        cv2.ellipse(frame, center, (55, 85), 0, 0, 360, (35, 45, 210), -1)
        if not cv2.imwrite(str(plate / f"{index:06d}.png"), frame):
            raise RuntimeError("Could not write synthetic source frame.")
        if index == 0:
            hint = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(hint, center, (50, 80), 0, 0, 360, 255, -1)
            cv2.imwrite(str(job / "first-frame-hint.png"), hint)
    command = [
        sys.executable, str(args.source.resolve() / "infer.py"), "--device", args.device,
        "--input", str(plate), "--output_dir", str(job / "result"),
        "--hint_first_frame", str(job / "first-frame-hint.png"),
        "--num_frames", str(frames), "--frame_dir_fps", "24", "--output_format", "exr",
        "--outputs", "alpha", "fg", "cutout", "--low_vram", "--hann_chunk", "4",
        "--hann_stride", "2", "--low_vram_foundation_chunk", "1",
        "--temp_dir", str(job / "temp"),
    ]
    report = {"engine": "v2", "device": args.device, "compute": compute, "command": command,
              "pretrained_inference_validated": False, "video_encoding_validated": False,
              "scope": "Six synthetic frames only; not real-footage quality or long-clip validation."}
    started = time.perf_counter()
    print(f"Running pretrained XPU inference. Log: {job / 'inference.log'}", flush=True)
    try:
        with (job / "inference.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=args.source.resolve(), check=True, stdout=log, stderr=subprocess.STDOUT)
        report["outputs"] = inspect_outputs(job / "result", frames, height, width)
        report["pretrained_inference_validated"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        (job / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Validation report: {job / 'validation.json'}", flush=True)


if __name__ == "__main__":
    main()
