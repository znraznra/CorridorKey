"""Version isolation, non-destructive source preparation and XPU dispatch."""

import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_v2_setup_is_separate_and_v1_stays_default(monkeypatch):
    bootstrap = module("ck2_bootstrap", "bootstrap.py")
    monkeypatch.setattr(bootstrap, "python312_command", lambda: ["python3.12"])
    original = bootstrap.build_plan("p600")
    arc = bootstrap.build_plan("arc", engine="v2", device="xpu:1", with_tests=True)
    assert original["runtime_file"].endswith("runtime.json")
    assert str(ROOT / "scripts/setup_gpu.py") in original["setup_command"]
    assert arc["runtime_file"].endswith("runtime-v2.json")
    assert arc["environment"] == str(ROOT / ".local/ck2/venv")
    assert str(ROOT / "scripts/ck2.py") in arc["setup_command"]
    assert "xpu:1" in arc["setup_command"] and "--with-tests" in arc["setup_command"]
    with pytest.raises(ValueError, match="v1 for P600"):
        bootstrap.build_plan("p600", engine="v2")
    with pytest.raises(ValueError, match="xpu:N"):
        bootstrap.build_plan("arc", engine="v2", device="cuda")


@pytest.fixture
def config():
    return {
        "engine": "v2", "python": "isolated-python", "profile": "arc", "device": "xpu:1",
        "source": "downloaded-ck2", "ffmpeg": "system-ffmpeg", "ffprobe": "system-ffprobe",
        "encoder": "h264_qsv",
    }


def test_v2_default_is_transparent_exr_with_selected_device(config):
    launcher = module("ck2_launch_exr", "run.py")
    command = launcher.build_command(config, "keyer", ["--input", "plate.mp4", "--output_dir", "job"])
    assert command[0] == "isolated-python"
    assert command[3] == str(Path("downloaded-ck2") / "infer.py")
    assert command[command.index("--device") + 1] == "xpu:1"
    assert command[command.index("--output_format") + 1] == "exr"
    assert "h264_qsv" not in command
    assert command[command.index("--temp_dir") + 1] == str(Path("job").resolve() / ".ck2-temp")


def test_video_preview_explicit_qsv_bitrate_and_no_crf(config):
    launcher = module("ck2_launch_video", "run.py")
    command = launcher.build_command(config, "keyer", ["--output_format=video", "--outputs", "checker"])
    assert command[command.index("--ffmpeg_codec") + 1] == "h264_qsv"
    assert command[command.index("--pix_fmt") + 1] == "nv12"
    assert command[command.index("--bitrate") + 1] == "12M"
    assert "--crf" not in command
    with pytest.raises(ValueError, match="without --crf"):
        launcher.build_command(config, "keyer", ["--output_format", "video", "--crf", "12"])
    with pytest.raises(ValueError, match="exr"):
        launcher.build_command(config, "keyer", ["--output_format", "video", "--outputs=alpha,cutout"])
    with pytest.raises(ValueError, match="exr"):
        launcher.build_command(config, "keyer", ["--output_format", "video", "--outputs", "CUTOUT"])
    with pytest.raises(ValueError, match="bootstrap"):
        launcher.build_command(config, "keyer", ["--device=cuda"])


def test_loading_v2_does_not_require_or_mutate_v1(tmp_path, monkeypatch, config):
    launcher = module("ck2_config", "run.py")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    (tmp_path / ".local").mkdir()
    (tmp_path / "integrations/ck2").mkdir(parents=True)
    (tmp_path / "integrations/ck2/xpu.patch").write_bytes(b"fixture patch")
    for name in ("python", "ffmpeg", "ffprobe"):
        (tmp_path / config[name]).write_bytes(b"fixture")
    (tmp_path / config["source"]).mkdir()
    (tmp_path / config["source"] / "infer.py").write_text("# fixture")
    config["patch_sha256"] = hashlib.sha256(b"fixture patch").hexdigest()
    (tmp_path / ".local/runtime-v2.json").write_text(json.dumps(config))
    v1 = tmp_path / ".local/runtime.json"
    v1.write_bytes(b"existing P600 config stays byte-identical")
    assert launcher.load_runtime("v2")["source"] == str(tmp_path / config["source"])
    assert v1.read_bytes() == b"existing P600 config stays byte-identical"
    broken = dict(config, device="cuda")
    (tmp_path / ".local/runtime-v2.json").write_text(json.dumps(broken))
    with pytest.raises(RuntimeError, match="XPU device"):
        launcher.load_runtime("v2")
    (tmp_path / ".local/runtime-v2.json").write_text(json.dumps(config))
    (tmp_path / "integrations/ck2/xpu.patch").write_bytes(b"changed patch")
    with pytest.raises(RuntimeError, match="patch changed"):
        launcher.load_runtime("v2")


def test_fetch_patch_rerun_and_conflicting_edits_are_preserved(tmp_path, monkeypatch):
    setup = module("ck2_setup", "scripts/ck2.py")
    origin = tmp_path / "origin"
    origin.mkdir()
    setup.git(origin, "init", "-q")
    (origin / "runtime.py").write_text("mode = 'cuda'\n")
    setup.git(origin, "add", "runtime.py")
    setup.git(origin, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    revision = setup.git(origin, "rev-parse", "HEAD").stdout.strip()
    integration = tmp_path / "integration"
    integration.mkdir()
    patch = "".join(difflib.unified_diff(
        ["mode = 'cuda'\n"], ["mode = 'xpu'\n"], fromfile="a/runtime.py", tofile="b/runtime.py"))
    (integration / "xpu.patch").write_text(patch)
    monkeypatch.setattr(setup, "INTEGRATION", integration)
    source = tmp_path / "local/source"
    setup.prepare_source(source, str(origin), revision)
    assert (source / "runtime.py").read_text() == "mode = 'xpu'\n"
    setup.prepare_source(source, str(origin), revision)
    (source / "runtime.py").write_text("mode = 'my-local-work'\n")
    with pytest.raises(RuntimeError, match="Preserve local edits"):
        setup.prepare_source(source, str(origin), revision)
    assert (source / "runtime.py").read_text() == "mode = 'my-local-work'\n"


def fake_xpu(monkeypatch):
    accelerator = module("ck2_accelerator", "integrations/ck2/accelerator.py")
    xpu = MagicMock()
    xpu.is_available.return_value = True
    xpu.device_count.return_value = 2
    xpu.current_device.return_value = 0
    monkeypatch.setattr(accelerator, "torch", SimpleNamespace(device=torch.device, xpu=xpu))
    return accelerator, xpu


def test_xpu_backend_never_accesses_cuda_and_retains_index(monkeypatch):
    accelerator, xpu = fake_xpu(monkeypatch)
    device = accelerator.require_device("xpu:1")
    assert device == torch.device("xpu:1")
    accelerator.empty_cache(device)
    accelerator.synchronize(device)
    xpu.set_device.assert_called_once_with(device)
    xpu.device.assert_called_once_with(device)
    xpu.empty_cache.assert_called_once_with()
    xpu.synchronize.assert_called_once_with(device)
    model = SimpleNamespace(_corridorkey_device="xpu:1", stage1=MagicMock())
    assert accelerator.runtime_device(model) == device
    model.stage1.parameters.assert_not_called()


def test_missing_or_invalid_xpu_fails_without_fallback(monkeypatch):
    accelerator, xpu = fake_xpu(monkeypatch)
    with pytest.raises(ValueError, match="index out of range"):
        accelerator.require_device("xpu:8")
    xpu.is_available.return_value = False
    with pytest.raises(RuntimeError, match="no fallback"):
        accelerator.require_device("xpu")
    with pytest.raises(ValueError, match="no CPU fallback"):
        accelerator.require_device("cpu")


def test_full_setup_refuses_host_platform_before_any_mutation(monkeypatch):
    setup = module("ck2_wrong_host", "scripts/ck2.py")
    monkeypatch.setattr(setup.sys, "platform", "win32")
    prepare = MagicMock()
    monkeypatch.setattr(setup, "prepare_source", prepare)
    with pytest.raises(RuntimeError, match="Linux"):
        setup.setup(SimpleNamespace(source_only=False))
    prepare.assert_not_called()


def test_p600_command_still_uses_original_cli():
    launcher = module("ck2_v1_regression", "run.py")
    command = launcher.build_command({"python": "p600-python", "device": "cuda"}, "keyer", ["run-inference"])
    assert command == ["p600-python", "-X", "utf8", str(ROOT / "corridorkey_cli.py"), "--device", "cuda",
                       "run-inference"]
