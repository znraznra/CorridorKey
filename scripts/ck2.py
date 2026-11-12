"""Provision the separate, experimental CorridorKey 2 Intel Arc runtime."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations/ck2"
LOCAL = ROOT / ".local/ck2"
UPSTREAM_URL = "https://github.com/nikopueringer/CorridorKey2.0.git"
REVISION = "485f5e9ab759ead6138cc990776184b59d6ea971"
WEIGHTS_REVISION = "cf44a8db3f024bb49fd59daa408848ccf05f7477"


def git(source, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(source), *args], check=check,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def prepare_source(source=None, url=UPSTREAM_URL, revision=REVISION):
    source = Path(source) if source is not None else LOCAL / "source"
    patch = INTEGRATION / "xpu.patch"
    patch_sha = hashlib.sha256(patch.read_bytes()).hexdigest()
    if not (source / ".git").exists():
        if source.exists() and any(source.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-Git source folder: {source}")
        source.mkdir(parents=True, exist_ok=True)
        git(source, "init", "-q")
        git(source, "remote", "add", "origin", url)
    head = git(source, "rev-parse", "--verify", "HEAD", check=False)
    if head.returncode != 0:
        if git(source, "status", "--porcelain").stdout.strip():
            raise RuntimeError("Uncommitted files in incomplete CK2 checkout; move it aside before retrying.")
        print(f"Fetching CorridorKey 2 source at {revision}...", flush=True)
        git(source, "fetch", "--depth", "1", "origin", revision)
        git(source, "checkout", "--detach", revision)
    elif head.stdout.strip() != revision:
        raise RuntimeError("CK2 source has a different revision; move it aside rather than resetting local work.")
    # Repeated setup is non-destructive. A changed or locally edited patch must
    # pass Git's checks; never reset, clean, or overwrite the downloaded tree.
    if git(source, "apply", "--reverse", "--check", str(patch), check=False).returncode != 0:
        check = git(source, "apply", "--check", str(patch), check=False)
        if check.returncode != 0:
            raise RuntimeError("CK2 patch does not apply cleanly. Preserve local edits and move the source "
                               f"folder aside before rerunning setup.\n{check.stderr}")
        git(source, "apply", str(patch))
    state = {"upstream": url, "revision": revision, "patch_sha256": patch_sha}
    (source.parent / "source-state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return source


def download_models(python, source):
    subprocess.run([
        str(python), str(source / "scripts/download_weights.py"),
        "--package_root", str(source), "--corridorkey_revision", WEIGHTS_REVISION,
    ], cwd=source, check=True)


def write_runtime(python, source, device, ffprobe=None):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = str(Path(ffprobe).resolve()) if ffprobe else shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("Install system FFmpeg including ffprobe before setup.")
    subprocess.run([ffprobe, "-v", "quiet", "-version"], check=True, stdout=subprocess.DEVNULL)
    sys.path.insert(0, str(ROOT / "scripts"))
    from configure_runtime import portable_path

    config = {
        "engine": "v2", "profile": "arc", "device": device, "experimental": True,
        "python": portable_path(python), "source": portable_path(source),
        "ffmpeg": portable_path(ffmpeg), "ffprobe": portable_path(ffprobe), "encoder": "h264_qsv",
        "upstream_revision": REVISION, "weights_revision": WEIGHTS_REVISION,
        "patch_sha256": hashlib.sha256((INTEGRATION / "xpu.patch").read_bytes()).hexdigest(),
        "pretrained_inference_validated": False, "encoding_hardware_validated": False,
    }
    destination = ROOT / ".local/runtime-v2.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def setup(args):
    if not args.source_only and (sys.platform != "linux" or sys.version_info[:2] != (3, 12)):
        raise RuntimeError("The CK2 Arc installer currently targets Linux with Python 3.12 only. "
                           "Use --source-only for patch review elsewhere; P600 stays on v1.")
    if args.device != "xpu" and not (args.device.startswith("xpu:") and args.device[4:].isdigit()):
        raise ValueError("CK2 setup accepts xpu or xpu:N only; no CUDA/CPU fallback.")
    if not args.source_only and (not shutil.which("ffmpeg") or not (args.ffprobe or shutil.which("ffprobe"))):
        raise RuntimeError("Install system FFmpeg including ffprobe before v2 bootstrap.")
    source = prepare_source()
    if args.source_only:
        print(f"Patched source only: {source}. No environment, model download or GPU validation.")
        return
    python = LOCAL / "venv/bin/python"
    if not python.exists():
        venv.create(python.parents[1], with_pip=True)

    def install(*packages):
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", *packages], check=True)

    install("torch==2.9.1+xpu", "torchvision==0.24.1+xpu", "--index-url", "https://download.pytorch.org/whl/xpu")
    constraints = LOCAL / "constraints.txt"
    constraints.write_text("torch==2.9.1+xpu\ntorchvision==0.24.1+xpu\n"
                           "numpy==1.26.4\nopencv-python==4.11.0.86\n", encoding="utf-8")
    install("-r", str(INTEGRATION / "requirements.txt"), "-c", str(constraints))
    if args.with_tests:
        install("pytest", "pytest-env")
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    # Fail on missing drivers/operators before downloading ~5 GB of models.
    subprocess.run([
        str(python), str(INTEGRATION / "check_xpu.py"), "--source", str(source),
        "--device", args.device, "--output", str(LOCAL / "compute-check.json"),
    ], cwd=source, check=True)
    if not args.skip_model:
        download_models(python, source)
    config = write_runtime(python, source, args.device, args.ffprobe)
    print(json.dumps(config, indent=2))
    print("Experimental CK2 Arc environment ready. Run a short pretrained inference sample next. "
          "The v1 environment and configuration were not modified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "download"))
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--with-tests", action="store_true")
    parser.add_argument("--ffprobe", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")
    if args.action == "setup":
        setup(args)
    else:
        sys.path.insert(0, str(ROOT))
        from run import load_runtime

        config = load_runtime("v2")
        download_models(config["python"], Path(config["source"]))


if __name__ == "__main__":
    main()
