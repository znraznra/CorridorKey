# CorridorKey 2: experimental Intel Arc backend

The original CorridorKey 1 backend remains the default, including the tested
Quadro P600 profile. CorridorKey 2 is an explicitly selected, experimental
backend for the Intel Arc B580 on Linux. B580 inference has not been executed
on the development machine; setup and the model smoke test must pass there
before treating it as working support. There is no automatic version, CPU,
CUDA or video-encoder fallback.

Local validation on 2026-11-12: the full suite passed 465 tests with one optional
MLX skip, and six isolated CK2 pipeline tests passed under PyTorch 2.9.1 CPU
using small stand-in models. Fresh pretrained P600 inference at internal 512
also passed against a CPU reference. This is engineering validation, not B580
execution evidence. Details: [validation record](ck2-validation-2026-11-12.json).

## Isolation and source

| Item | Version 1 | Version 2 |
| --- | --- | --- |
| Source | Existing repository engine | Pinned upstream checkout under `.local/ck2/source` |
| Environment | `.venv` | `.local/ck2/venv` |
| Runtime configuration | `.local/runtime.json` | `.local/runtime-v2.json` |
| PyTorch | 2.8.0 with selected profile | 2.9.1+xpu, torchvision 0.24.1+xpu |
| NumPy / OpenCV | Existing environment | 1.26.4 / 4.11.0.86 |
| P600 | Retained FP32 eager CUDA path | Not supported by this integration |

Setup fetches [upstream CorridorKey 2.0](https://github.com/nikopueringer/CorridorKey2.0)
at `485f5e9ab759ead6138cc990776184b59d6ea971`, applies the checked-in
`integrations/ck2/xpu.patch`, and retains upstream's licence and third-party
notices. The CK2 model revision is
`cf44a8db3f024bb49fd59daa408848ccf05f7477`; upstream also pins its three
foundation models. Approximately 5 GB of model downloads are needed, plus
packages, caches, source and output storage. Everything fetched stays outside Git.

The patch adds explicit XPU device selection, BF16 autocasting, indexed device
transfers, cache handling and synchronization. Model initialization takes place
on CPU before staged GPU transfers. Default XPU processing uses staged low-VRAM
inference, a Hann window of up to four frames with half-window stride, and one
frame per foundation chunk. This preserves native output dimensions, but the
shorter temporal window can affect temporal quality. Larger windows require
on-device memory and quality checks. The upstream CUDA memory estimate table is
not used for XPU. Eager PyTorch/native SDPA is used; no mandatory CUDA-only
FlashAttention or xformers installation is added.

The patch also propagates output-worker failures through bounded queues so an
EXR-write or encoder failure does not leave the producer waiting indefinitely.

## Arch Linux / B580 setup

First push these source changes to your own GitHub repository. On a fresh Arch
device, replace the example repository URL:

```bash
sudo pacman -Syu --needed git uv ffmpeg linux-firmware-intel \
  intel-compute-runtime level-zero-loader intel-media-driver vpl-gpu-rt
sudo usermod -aG render "$USER"
sudo reboot
```

After reboot:

```bash
mkdir -p "$HOME/Projects"
cd "$HOME/Projects"
git clone https://github.com/YOUR-USER/YOUR-REPO.git CorridorKey
cd CorridorKey
uv python install 3.12
uv run --no-project --python 3.12 python bootstrap.py --engine v2 --profile arc
```

For an existing clone, use `git pull --ff-only`, then the bootstrap command.
Select `--device xpu:1` during bootstrap if the B580 is device 1 rather than the
default Intel GPU. Verify the reported GPU name; an integrated Intel GPU may
also be enumerated. Arch is not among PyTorch's listed validated distributions.
Drivers, kernel and device permissions remain system prerequisites.

Bootstrap runs real XPU BF16 convolution, matmul, SDPA, RMSNorm and area-resize
checks before downloading models. A missing GPU or failed operator stops setup.
These checks do not prove pretrained-model correctness or video encoding.

Options include `--dry-run`, `--with-tests`, `--skip-model` and
`--ffprobe /absolute/path/to/ffprobe`. Download deferred weights later with:

```bash
.local/ck2/venv/bin/python scripts/ck2.py download
```

Do not run `uv sync` or install CK2 requirements into the v1 `.venv`.
For source/patch review without provisioning a GPU environment, run
`python scripts/ck2.py setup --source-only` on Windows or Linux.
Repeated setup preserves local source edits and stops on patch conflicts;
it never runs `git reset` or `git clean`. If a patch or pinned revision changes,
move the old `.local/ck2/source` folder aside before rerunning setup. This keeps
the previous source and downloaded assets available for inspection.

## Validate the B580

```bash
.local/ck2/venv/bin/python run.py --engine v2 check
.local/ck2/venv/bin/python run.py --engine v2 verify \
  --output Output/ck2-b580-smoke
```

The second command runs the actual pretrained model on six moving synthetic
frames with a first-frame hint, overlapping temporal windows and staged
foundation processing. It checks the frame count, dimensions, channels, finite
EXR values and alpha range/variation. Inspect `validation.json`, `inference.log`
and the output sequences. It requires a fresh output directory and preserves
failed-run evidence. Synthetic success does not establish real-footage edge
quality, long-clip reliability, performance or QSV encoding. Runtime setup
metadata remains conservative; validation evidence is stored separately.

## Process your own plate

Start with a short sample; replace paths with absolute paths on your device:

```bash
.local/ck2/venv/bin/python run.py --engine v2 keyer \
  --input /absolute/path/plate.mp4 \
  --output_dir /absolute/path/job/sample \
  --num_frames 12 --outputs alpha fg cutout
```

The launcher defaults to **EXR sequences**. CK2 can run without a hint, with
`--hint_first_frame /absolute/path/hint.png`, or with
`--hint_video /absolute/path/aligned-hints.mp4` (a frame directory also works).
Use visible grayscale hints: white subject, black background, soft gray edges.
Hints must match the source's orientation, dimensions and frame timing.
Resolve preparation remains in the skill's Resolve reference.

After inspecting the sample, remove `--num_frames 12` for the whole clip.
Native output dimensions are preserved. Each job gets its own temporary cache
under its output directory unless `--temp_dir` is supplied. Do not run two jobs
against the same output/cache folder. Low-VRAM processing needs local disk for
feature caches proportional to clip length and resolution.

Version 2 output conventions differ from v1:

- `*_alpha_exr`: linear scalar alpha, clamped to [0,1].
- `*_fg_exr`: straight scene-linear RGB, preserving floating-point range.
- `*_cutout_exr`: straight scene-linear RGBA, with the upstream despill applied.
- `*_checker_exr` when requested: an opaque scene-linear checkerboard composite.

Treat CK2 cutout alpha as **straight** in Resolve. V1's `Processed` sequence is
premultiplied; do not apply that convention to CK2. EXRs contain no audio.
The original plate remains the audio source. Upstream encoded previews are
silent as well; audio preservation requires an explicit subsequent mux/export.

For a green composite, pass the CK2 alpha and FG EXR directories to the shared
`run.py --engine v2 green` helper with `--fg-space linear`. That helper
expects CFR frame alignment; ordinary-background footage still needs a suitable
matte and must not be represented as validated arbitrary-background removal.

For an opaque encoded preview, explicitly request `--output_format video
--outputs checker`. The launcher then selects `h264_qsv`, `nv12` and an explicit
12M bitrate by default; it does not select a software fallback. Encoder/device
support and playback must be checked on the B580. An Intel iGPU may be selected
by FFmpeg independently of PyTorch's XPU device index; successful encoding alone
does not prove B580 media-engine use. `--bitrate` can override the default.
Transparent video cutout is rejected by this launcher: upstream forces software
ProRes for it, so use EXR for transparency. Source color management still needs
review, especially for log/HDR footage. Produce the requested local review
preview and preserve source audio deliberately in the delivery workflow.

## P600 usage stays on version 1

Existing commands remain valid:

```text
python bootstrap.py --profile p600
python run.py check
python run.py keyer run-inference --help
```

`--engine v1` is optional. V2 bootstrap refuses the P600 profile and does not
change the P600 environment, checkpoints or runtime configuration.

## Licence and distribution

The [Corridor Key Licence v2](https://github.com/nikopueringer/CorridorKey2.0/blob/master/LICENSE)
permits local modification and commercial processing subject to its additional
terms. Released derivatives must be freely accessible, retain attribution and
follow share-alike terms. Paid inference, commercial software integration and
repackaging/resale are restricted; entities exceeding the stated US$20 million
group revenue threshold require paid seats for users. Consult the complete
licence before distribution or integration. Foundation models retain their own
terms in upstream `THIRD_PARTY_NOTICES.md`. Compatibility patches follow that
same Corridor Key Licence v2. This integration does not supply the paid Resolve
plugin, its UI, or an unrestricted licence to redistribute the tool as a product.
