# CorridorKey: Quadro P600 and Intel Arc profiles

Historical initial setup and measurements. For the current portable bootstrap,
harness integrations and support levels, start with [README.md](README.md) and
[PORTABLE.md](PORTABLE.md). The original launchers remain available but do not
configure the new portable runtime file.

Local adaptation of upstream commit `97e55a453060745bead1befd293f6e523c4b845c`.
Prepared for Arnold Libelo. The original CorridorKey name and licence are retained.

These profiles run the **CorridorKey keyer** using supplied coarse alpha hints.
GVM, VideoMaMa and BiRefNet hint generation have not been validated on either
profile. Do not select the large optional hint models on the P600.

## Windows / Quadro P600

Python 3.12 is required. Run `Setup_P600.bat` to install into this folder's
`.venv`; it does not change global Python or install a system CUDA toolkit.
The installer selects **PyTorch 2.8.0 + CUDA 12.6** and torchvision 0.23.0.
It pins OpenCV 4.12.0.88 because the OpenCV 5 wheel tested here lacks EXR writing.
The upstream CUDA 12.8 installer excludes Pascal and must not be used here.

The keyer selects FP32 and eager execution for CUDA GPUs below capability 7.0.
FP16 has low arithmetic throughput on the P600. Eager execution also avoids
unsupported Triton compilation. A CUDA-enabled torch build being installed is
not proof of compatibility: the installer runs actual GPU convolution/matmul.

Check the device:

```powershell
.venv\Scripts\python.exe scripts\gpu_check.py --device cuda
```

Use existing CorridorKey shot folders, each containing matching `Input` and
`AlphaHint` sequences or videos. For example:

```text
ClipsForInference/
  shot01/
    Input/frame_0000.png
    AlphaHint/frame_0000.png
```

Run a fully specified batch at the recommended 1024 internal resolution:

```powershell
.\Run_P600.bat run-inference --backend torch --image-size 1024 --srgb --despill 5 --no-despeckle --refiner 1 --comp --cpu-post --no-tile --screen-color green
```

Outputs retain the input dimensions and are written to each shot's `Output`
folder. Internal 512/1024 processing followed by scaling back up does not recover
the native 2048 model's fine detail. Use 512 for faster drafts. Source dimensions
also affect memory use; the included tests use 1280x720 input.

### Measured on this P600 on 2026-11-12

The NVIDIA driver reports 4 GiB dedicated memory, driver 582.78, capability 6.1.
FP32 eager inference, official green checkpoint, CPU post-processing, supplied
synthetic hints; timings exclude model loading/warmup and include post-processing.

| Internal size | Mean seconds/frame | Peak allocated | Peak reserved | Scope |
| --- | ---: | ---: | ---: | --- |
| 512 | 0.97 | 0.50 GiB | 0.69 GiB | 3 timed frames; CPU comparison passed |
| 1024 | 3.26 | 1.17 GiB | 1.83 GiB | 3 timed frames |
| 2048 | 34.30 | 3.86 GiB | 6.35 GiB | 1 timed frame after warmup |

These are short synthetic smoke measurements, not real-footage throughput claims.
At 2048 the allocator reserved more memory than dedicated VRAM; Windows memory
management allowed this short run to complete. It does **not** establish a
reliable 4 GiB fit or long-video stability. Prefer 1024 on this workstation.
The 512 CPU/GPU mean absolute errors were approximately 7.6e-9 for alpha and
2.1e-8 for foreground. JSON reports and synthetic comparison images are under
`Output/p600-512`, `Output/p600-1024`, and `Output/p600-2048` in the tested folder.

The regression suite passed 438 tests across the regular and GPU-marked runs,
with one optional MLX test skipped. Real trained-model GPU verification is
recorded separately by `scripts/verify_inference.py`. The normal Windows batch
launcher also processed three synthetic input/hint pairs at 1024 and wrote all
four output passes. A three-second synthetic 480p MP4 preview is available at
`Output/p600-1024/preview-480p.mp4`; it repeats each of the three inferred frames
for one second and is not a real-time throughput demonstration.

`Run_P600.bat --help` and `Run_P600.bat run-inference --help` show other options.
These launchers use `.venv` directly. Do not run `uv sync --extra cuda` in this
environment: that would restore the upstream CUDA 12.8 wheel selection.

## Arch Linux / Intel Arc B580

**Hardware validation is pending on the B580.** This Windows workstation has
no Arc GPU. The XPU routing and error handling are unit-tested with simulated
hardware; that does not establish driver/operator support on the other device.

Use a current kernel/firmware supporting the B580, working `/dev/dri/render*`
permissions, and Intel's Level Zero compute runtime. Arch's relevant packages
are `intel-compute-runtime` and `level-zero-loader`:

- https://archlinux.org/packages/extra/x86_64/intel-compute-runtime/
- https://archlinux.org/packages/extra/x86_64/level-zero-loader/
- https://docs.pytorch.org/docs/main/notes/get_start_xpu.html

Arch is not in PyTorch's validated Ubuntu distribution matrix. These instructions
do not change drivers or system packages automatically. The setup script needs
Python 3.12 or `uv` (which can supply Python 3.12 without replacing Arch's system
Python). Override the executable with `PYTHON312` if necessary.

Copy/extract the source folder locally on the Arch device, then:

```bash
bash setup_arc_linux.sh
.venv/bin/python scripts/gpu_check.py --device xpu --output Output/arc-device.json
.venv/bin/python scripts/verify_inference.py --device xpu --image-size 512 --frames 3 --compare-cpu --output Output/arc-512
```

Setup installs the official PyTorch 2.8.0 XPU wheels and uses FP16 eager inference.
It fails if Intel GPU execution is unavailable; an explicit `xpu` request never
silently falls back to CPU. The official keyer weights download on first run
(about 399 MB), or copy the local `.safetensors` checkpoint into
`CorridorKeyModule/checkpoints/` before running.

Inspect multiple Intel devices if necessary:

```bash
.venv/bin/python -c 'import torch; print([(i, torch.xpu.get_device_name(i)) for i in range(torch.xpu.device_count())])'
CORRIDORKEY_DEVICE=xpu:0 bash run_arc_linux.sh run-inference --backend torch --image-size 512 --srgb --despill 5 --no-despeckle --refiner 1 --comp --cpu-post --no-tile --screen-color green
```

Use the B580's actual index if it differs from zero. After a successful 512 test,
repeat at 1024, then assess 2048 on the actual card. Memory usage and throughput
cannot be inferred from the P600 results.

## Reproduce validation

```powershell
.venv\Scripts\python.exe scripts\setup_gpu.py --profile p600 --with-tests
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\verify_inference.py --device cuda --image-size 512 --frames 3 --compare-cpu --output Output\p600-512
```

`verification.json` records the actual model input device, checkpoint hash,
processing precision, warmup, frame timings, peak PyTorch memory, finite output
checks and optional CPU comparison. Its generated images are synthetic test
fixtures; real green-screen footage is still needed to assess production quality.
No retraining, cloud processing or uploads are involved.
