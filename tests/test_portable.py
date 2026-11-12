"""Relocation and dispatch invariants for the one-checkout workflow."""

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_plan_uses_new_checkout_location(tmp_path, monkeypatch):
    bootstrap = module_at("portable_bootstrap", ROOT / "bootstrap.py")
    relocated = tmp_path / "different folder" / "CorridorKey"
    monkeypatch.setattr(bootstrap, "ROOT", relocated)
    monkeypatch.setattr(bootstrap, "python312_command", lambda: ["python3.12"])
    plan = bootstrap.build_plan("arc", with_tests=True)
    assert plan["device"] == "xpu"
    assert str(relocated / "scripts/setup_gpu.py") in plan["setup_command"]
    assert "--with-tests" in plan["setup_command"]
    assert plan["skill"] == str(relocated / ".agents/skills/corridorkey/SKILL.md")
    assert not relocated.exists(), "Planning must not create/install anything"


def test_relative_runtime_paths_follow_relocated_clone(tmp_path, monkeypatch):
    launcher = module_at("portable_launcher", ROOT / "run.py")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    config = {
        "profile": "p600",
        "device": "cuda",
        "encoder": "h264_nvenc",
        "python": ".venv/python",
        "ffmpeg": ".local/ffmpeg",
        "ffprobe": ".local/ffprobe",
    }
    for key in ("python", "ffmpeg", "ffprobe"):
        file = tmp_path / config[key]
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"test fixture, not an executable")
    (tmp_path / ".local/runtime.json").write_text(json.dumps(config))
    loaded = launcher.load_runtime()
    assert loaded["python"] == str(tmp_path / ".venv/python")
    assert loaded["ffmpeg"] == str(tmp_path / ".local/ffmpeg")


def test_missing_tool_stops_instead_of_using_another_profile(tmp_path, monkeypatch):
    launcher = module_at("portable_missing", ROOT / "run.py")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    (tmp_path / ".local").mkdir()
    (tmp_path / ".local/runtime.json").write_text(json.dumps({"python": "missing"}))
    with pytest.raises(RuntimeError, match="rerun bootstrap|Rerun bootstrap"):
        launcher.load_runtime()


@pytest.mark.parametrize("device,encoder", [("cuda", "h264_nvenc"), ("xpu", "h264_qsv"), ("cpu", "libx264")])
def test_green_dispatch_keeps_profile_encoder(device, encoder):
    launcher = module_at("portable_dispatch", ROOT / "run.py")
    config = {
        "python": "env-python",
        "device": device,
        "encoder": encoder,
        "ffmpeg": "local-ffmpeg",
        "ffprobe": "local-ffprobe",
    }
    command = launcher.build_command(config, "green", ["--video", "source.mp4"])
    assert command[command.index("--encoder") + 1] == encoder
    assert command[command.index("--ffmpeg") + 1] == "local-ffmpeg"
    assert command[command.index("--ffprobe") + 1] == "local-ffprobe"


def test_explicit_encoder_is_not_duplicated_or_overridden():
    launcher = module_at("portable_override", ROOT / "run.py")
    config = {
        "python": "env-python",
        "device": "xpu",
        "encoder": "h264_qsv",
        "ffmpeg": "local-ffmpeg",
        "ffprobe": "local-ffprobe",
    }
    command = launcher.build_command(config, "green", ["--encoder=libx264"])
    assert "--encoder=libx264" in command
    assert "h264_qsv" not in command


def test_cpu_doctor_runs_real_cpu_operations():
    doctor = module_at("portable_doctor", ROOT / "scripts/gpu_check.py")
    report = doctor.inspect_gpu("cpu")
    assert report["device"] == "cpu"
    assert report["cpu_convolution_and_matmul"] == "passed"
    assert "gpu_convolution_and_matmul" not in report


def test_saved_internal_paths_are_relative(tmp_path, monkeypatch):
    configure = module_at("portable_configure", ROOT / "scripts/configure_runtime.py")
    monkeypatch.setattr(configure, "ROOT", tmp_path)
    assert configure.portable_path(tmp_path / ".venv/python") == ".venv/python"


def probe_download_fixture(monkeypatch, tmp_path, checksum_valid=True):
    configure = module_at("portable_download", ROOT / "scripts/configure_runtime.py")
    monkeypatch.setattr(configure, "ROOT", tmp_path)
    monkeypatch.setattr(configure, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(configure.shutil, "which", lambda _: None)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("build/bin/ffprobe.exe", b"synthetic executable fixture")
        bundle.writestr("../../outside.txt", b"must not be extracted")
    payload = archive.getvalue()
    digest = hashlib.sha256(payload).hexdigest() if checksum_valid else "0" * 64
    monkeypatch.setattr(configure.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(digest.encode()))
    monkeypatch.setattr(configure.urllib.request, "urlretrieve", lambda url, path: Path(path).write_bytes(payload))
    return configure


def test_probe_download_checks_hash_and_extracts_only_named_executable(tmp_path, monkeypatch):
    configure = probe_download_fixture(monkeypatch, tmp_path)
    result = configure.find_probe()
    assert result.read_bytes() == b"synthetic executable fixture"
    assert not (tmp_path.parent / "outside.txt").exists()
    assert not (tmp_path / ".local/tools/ffmpeg-download.zip").exists()


def test_probe_bad_checksum_never_extracts_executable(tmp_path, monkeypatch):
    configure = probe_download_fixture(monkeypatch, tmp_path, checksum_valid=False)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        configure.find_probe()
    assert not (tmp_path / ".local/tools/ffprobe.exe").exists()


def test_skill_package_has_manifest_helpers_and_no_runtime(tmp_path):
    packager = module_at("portable_packager", ROOT / "scripts/package_skill.py")
    result = packager.package(tmp_path / "skill.zip")
    with zipfile.ZipFile(result["path"]) as bundle:
        entries = bundle.namelist()
        assert "corridorkey/SKILL.md" in entries
        assert "corridorkey/scripts/compose_green.py" in entries
        assert "corridorkey/references/runtime-portable.md" in entries
        assert not any(".venv" in entry or entry.endswith(".safetensors") for entry in entries)
