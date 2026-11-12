"""Discover local media tools and save machine-only settings for this checkout."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILES = {
    "p600": ("cuda", "h264_nvenc"),
    "cuda": ("cuda", "h264_nvenc"),
    "arc": ("xpu", "h264_qsv"),
    "cpu": ("cpu", "libx264"),
}
WINDOWS_TOOLS_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


def find_probe(explicit=None):
    if explicit:
        candidate = Path(explicit).resolve()
        if not candidate.is_file():
            raise RuntimeError(f"ffprobe not found: {candidate}")
        return candidate
    local = ROOT / ".local/tools/ffprobe.exe"
    if local.is_file():
        return local
    on_path = shutil.which("ffprobe")
    if on_path:
        return Path(on_path)
    if os.name != "nt":
        raise RuntimeError("Install FFmpeg/ffprobe using your OS package manager, then rerun bootstrap.")
    # Fetch the published checksum and extract only ffprobe. Never extract an
    # archive tree or install into a global directory/PATH.
    local.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(WINDOWS_TOOLS_URL + ".sha256", timeout=60) as response:
        checksum_text = response.read().decode()
    matches = re.findall(r"\b[0-9a-fA-F]{64}\b", checksum_text)
    if len(matches) != 1:
        raise RuntimeError("Could not parse the published FFmpeg package SHA-256.")
    archive = local.parent / "ffmpeg-download.zip"
    print("Downloading Windows ffprobe tools (about 110 MB); verifying published SHA-256...", flush=True)
    urllib.request.urlretrieve(WINDOWS_TOOLS_URL, archive)
    with archive.open("rb") as archive_data:
        actual = hashlib.file_digest(archive_data, "sha256").hexdigest()
    if actual.lower() != matches[0].lower():
        raise RuntimeError("FFmpeg archive checksum mismatch; no executable extracted. Rerun bootstrap.")
    temporary = local.with_suffix(".exe.tmp")
    with zipfile.ZipFile(archive) as bundle:
        candidates = [entry for entry in bundle.infolist() if entry.filename.endswith("/bin/ffprobe.exe")]
        if len(candidates) != 1:
            raise RuntimeError("Expected one ffprobe executable in the verified archive.")
        with bundle.open(candidates[0]) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
    temporary.replace(local)
    (local.parent / "ffprobe-source.json").write_text(
        json.dumps({"url": WINDOWS_TOOLS_URL, "archive_sha256": actual}, indent=2) + "\n", encoding="utf-8"
    )
    archive.unlink()
    return local


def portable_path(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    from imageio_ffmpeg import get_ffmpeg_exe

    device, encoder = PROFILES[args.profile]
    # The pinned imageio Windows binary is the one used in the P600 validation.
    # Linux's distro binary normally supplies Arc's QSV encoder.
    ffmpeg = Path((shutil.which("ffmpeg") if os.name != "nt" else None) or get_ffmpeg_exe())
    ffprobe = find_probe(args.ffprobe)
    subprocess.run([str(ffprobe), "-v", "quiet", "-version"], stdout=subprocess.DEVNULL, check=True)
    encoders = subprocess.check_output([str(ffmpeg), "-hide_banner", "-encoders"], text=True, stderr=subprocess.STDOUT)
    if not re.search(r"\b" + re.escape(encoder) + r"\b", encoders):
        raise RuntimeError(f"FFmpeg lacks {encoder}. Install a build with this encoder; no fallback selected.")
    if not args.skip_model:
        from CorridorKeyModule.backend import TORCH_EXT, _discover_checkpoint

        os.chdir(ROOT)
        _discover_checkpoint(TORCH_EXT, screen_color="green")
    config = {
        "profile": args.profile,
        "device": device,
        "encoder": encoder,
        "python": portable_path(sys.executable),
        "ffmpeg": portable_path(ffmpeg),
        "ffprobe": portable_path(ffprobe),
        "encoding_hardware_validated": False,
    }
    destination = ROOT / ".local/runtime.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2))
    print("Compute checks passed during setup; validate a short video export on this device before delivery.")


if __name__ == "__main__":
    main()
