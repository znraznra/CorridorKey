"""Install an isolated compute profile without changing global Python."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("p600", "arc", "cuda", "cpu"), required=True)
    parser.add_argument("--with-tests", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python 3.12 to create the pinned GPU environment.")
    env_dir = ROOT / ".venv"
    python = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        venv.create(env_dir, with_pip=True)
    index = {"p600": "cu126", "arc": "xpu", "cuda": "cu128", "cpu": "cpu"}[args.profile]

    def pip_install(*packages, **kwargs):
        subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", *packages],
            check=True,
            **kwargs,
        )

    # Use separate indexes explicitly. Never install CUDA wheels on the Arc
    # device, or cu128 (which omits Pascal) on the P600.
    torch_pin = f"torch==2.8.0+{index}"
    vision_pin = f"torchvision==0.23.0+{index}"
    pip_install(torch_pin, vision_pin, "--index-url", f"https://download.pytorch.org/whl/{index}")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = [
        dep
        for dep in project["project"]["dependencies"]
        if dep.split("=")[0].strip() not in ("torch", "torchvision", "triton-windows")
    ]
    # Eager execution needs no Triton. Keep the original project/uv lock intact
    # for other backends; these profiles run directly through .venv Python.
    constraints = env_dir / "gpu-constraints.txt"
    constraints.write_text(
        f"{torch_pin}\n{vision_pin}\nopencv-python==4.12.0.88\nopencv-python-headless==99999\n",
        encoding="utf-8",
    )
    pip_install(*dependencies, "--constraint", str(constraints))
    pip_install("imageio-ffmpeg==0.6.0")
    if args.with_tests:
        pip_install(*project["dependency-groups"]["dev"])
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    subprocess.run(
        [
            str(python),
            str(ROOT / "scripts/gpu_check.py"),
            "--device",
            {"p600": "cuda", "arc": "xpu", "cuda": "cuda", "cpu": "cpu"}[args.profile],
        ],
        cwd=ROOT,
        check=True,
    )
    (env_dir / "gpu-profile.json").write_text(json.dumps({"profile": args.profile, "index": index}), encoding="utf-8")
    print(f"GPU environment ready: {python}", flush=True)


if __name__ == "__main__":
    main()
