"""Optional patched-upstream integration tests, using tiny CPU stand-ins for weights.

Run `python scripts/ck2.py setup --source-only` first. These checks exercise the
real streaming/offload/EXR code; they do not validate XPU or pretrained models.
"""

import importlib
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def runtime(monkeypatch):
    source = Path(os.environ.get("CORRIDORKEY2_SOURCE", ROOT / ".local/ck2/source"))
    if not (source / "corridorkey/accelerator.py").is_file():
        pytest.skip("Fetch patched CK2 with scripts/ck2.py setup --source-only first")
    monkeypatch.syspath_prepend(str(source))
    module = importlib.import_module("corridorkey.runtime")
    monkeypatch.setattr(module, "empty_cache", lambda device: None)
    monkeypatch.setattr(module, "synchronize", lambda device: None)
    return module


class TinyRvm(torch.nn.Module):
    embed_dim = 2

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, rgb, state=None):
        batch, time, _, height, width = rgb.shape
        return torch.ones(batch, time, 2, height // 16, width // 16), state


class TinyFoundations(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rvm = TinyRvm()
        self.register_buffer("rvm_mean", torch.zeros(2))
        self.register_buffer("rvm_std", torch.ones(2))

    def forward(self, rgb_cradio, rgb_moge):
        count, _, height, width = rgb_cradio.shape
        features = torch.ones(count, 2, height // 16, width // 16)
        return {"cradio": features, "moge": features}


class TinyStage1(torch.nn.Module):
    d_16 = 2
    cradio_dim = 2
    moge_dim = 2

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, rgb, hint, moge_feat, **kwargs):
        batch, time, _, height, width = rgb.shape
        small = functional.interpolate(rgb.flatten(0, 1).float(), (height // 16, width // 16))
        return {"baton": small[:, :2].reshape(batch, time, 2, height // 16, width // 16),
                "aux": small[:, :1].reshape(batch, time, 1, height // 16, width // 16)}


class TinyStage2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, rgb, baton):
        alpha = (rgb[:, :, 0:1] > rgb[:, :, 1:2]).float()
        return torch.cat((alpha, rgb.float()), dim=2)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._corridorkey_device = "cpu"
        self.foundations = TinyFoundations()
        self.stage1 = TinyStage1()
        self.stage2 = TinyStage2()


def test_xpu_auto_profile_does_not_query_cuda(runtime, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA memory estimates cannot be used for XPU")

    monkeypatch.setattr(torch.cuda, "mem_get_info", forbidden)
    cfg = runtime.InferenceConfig(auto_vram=True)
    runtime.apply_auto_vram(cfg, (1080, 1920), 30, "xpu:1")
    assert (cfg.hann_chunk, cfg.hann_stride, cfg.low_vram) == (4, 2, True)


@pytest.mark.parametrize("low_vram", [False, True])
def test_real_streaming_pipeline_preserves_frames_and_exr_alpha(runtime, low_vram, tmp_path):
    import cv2

    torch.set_num_threads(2)
    source = tmp_path / "plate"
    source.mkdir()
    expected = []
    for index in range(6):
        frame = np.zeros((128, 128, 3), np.uint8)
        frame[:] = (20, 180, 20)
        frame[20:100, 30 + index:80 + index] = (20, 30, 200)
        expected.append((frame[:, :, 2] > frame[:, :, 1]).astype(np.float32))
        assert cv2.imwrite(str(source / f"{index:06d}.png"), frame)
    cfg = runtime.InferenceConfig(
        hann_chunk=4, hann_stride=2, low_vram=low_vram, num_frames=6,
        low_vram_foundation_chunk=1, outputs={"alpha", "fg", "cutout"},
        temp_dir=tmp_path / "cache", output_format="exr",
    )
    output = tmp_path / "output"
    runtime.run_inference(TinyModel(), str(source), str(output), cfg, runtime.WriterConfig(),
                          initial_hint_path=None, hint_video_path=None)
    files = sorted((output / "plate_alpha_exr").glob("*.exr"))
    assert len(files) == 6
    for file, alpha in zip(files, expected, strict=True):
        np.testing.assert_array_equal(cv2.imread(str(file), cv2.IMREAD_UNCHANGED), alpha)
    for kind, channels in (("fg", 3), ("cutout", 4)):
        files = sorted((output / f"plate_{kind}_exr").glob("*.exr"))
        assert len(files) == 6
        for file in files:
            frame = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
            assert frame.shape == (128, 128, channels)
            assert np.isfinite(frame).all()
    assert not list((tmp_path / "cache").glob("*.f16"))
    assert not list((tmp_path / "cache").glob("*.f32"))


def test_smoke_validator_rejects_missing_or_corrupt_exr(tmp_path):
    sys.path.insert(0, str(ROOT / "integrations/ck2"))
    verify = importlib.import_module("verify")
    with pytest.raises(RuntimeError, match="Expected one alpha"):
        verify.inspect_outputs(tmp_path, 6, 256, 384)


def test_failed_exr_worker_does_not_block_producer_or_release(runtime, tmp_path, monkeypatch):
    from corridorkey import video_io

    monkeypatch.setattr(video_io.cv2, "imwrite", lambda *args, **kwargs: False)
    writer = video_io.ExrSequenceWriter(str(tmp_path / "bad-output"), 24, (4, 4), maxsize=1)
    done = threading.Event()
    errors = []

    def write_until_failure():
        try:
            for _ in range(20):
                writer.write_many(np.ones((1, 4, 4, 3), np.float32))
        except RuntimeError as error:
            errors.append(str(error))
        finally:
            try:
                writer.release()
            except RuntimeError as error:
                errors.append(str(error))
            done.set()

    threading.Thread(target=write_until_failure, daemon=True).start()
    assert done.wait(3), "Writer failure must propagate rather than hanging on a full queue"
    assert errors and all("cv2.imwrite failed" in error for error in errors)


def test_smoke_validator_reads_actual_exr_and_detects_nan(tmp_path):
    import cv2

    sys.path.insert(0, str(ROOT / "integrations/ck2"))
    verify = importlib.import_module("verify")
    for kind, channels in (("alpha", 1), ("fg", 3), ("cutout", 4)):
        folder = tmp_path / f"plate_{kind}_exr"
        folder.mkdir()
        for index in range(2):
            frame = np.ones((8, 8, channels), np.float32)
            if kind == "alpha":
                frame[:4] = 0
            assert cv2.imwrite(str(folder / f"{index}.exr"), frame)
    report = verify.inspect_outputs(tmp_path, 2, 8, 8)
    assert report["cutout"]["shape"] == [8, 8, 4]
    assert cv2.imwrite(str(tmp_path / "plate_alpha_exr/0.exr"), np.full((8, 8), np.nan, np.float32))
    with pytest.raises(RuntimeError, match="finite"):
        verify.inspect_outputs(tmp_path, 2, 8, 8)
