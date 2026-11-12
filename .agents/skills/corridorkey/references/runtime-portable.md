# Portable runtime

Find this checkout by walking upward from this reference until both
`bootstrap.py` and `CorridorKeyModule/` exist. A user-scoped copy can instead use
an explicit `CORRIDORKEY_ROOT` or a user-provided checkout path. Never assume a
drive letter, username, another project's node_modules or a clone location.

Read `PORTABLE.md` for setup. Run the keyer from the checkout directory because
upstream checkpoint discovery still uses relative paths. `run.py` sets that
working directory automatically.

## Runtime selection

`python bootstrap.py --profile p600|arc|cuda|cpu` provisions `.venv` with Python
3.12 and writes ignored, machine-specific `.local/runtime.json`. Resolve its
relative interpreter/tool paths against this checkout. Each device needs its
own bootstrap; do not commit or copy runtime.json between machines.

- `p600`: torch 2.8.0+cu126 with sm_61; FP32 eager; NVENC export candidate.
- `arc`: torch 2.8.0+xpu; FP16 eager; QSV export candidate. B580/Arch hardware
  validation remains pending. Install suitable Intel drivers separately.
- `cuda`: torch 2.8.0+cu128 for compatible newer NVIDIA cards, not Pascal.
- `cpu`: explicit CPU inference and libx264 export, slower than suitable GPUs.
  This is a chosen profile, not a silent fallback from failed GPU execution.

OpenCV is pinned to 4.12.0.88 because the tested OpenCV 5 wheel lacked EXR writing.
Do not use upstream `uv sync --extra cuda` to maintain these environments: it
can replace the chosen Torch build. Use the profile bootstrap instead.

`python run.py check` performs real convolution and matrix operations on the
selected device. This does not validate video encoding. Inspect a short export
and its encoder diagnostics on each new device.

## Helpers

Run from the checkout with actual absolute media paths and fresh output names:

```text
python run.py hint --video /path/to/plate.mp4 --output /path/to/job/AlphaHint --screen green
python run.py green --video /path/to/plate.mp4 --matte /path/to/job/Matte --output /path/to/job/green-preview.mp4 --height 480 --max-frames 24 --bitrate 4M
```

`green` obtains explicit FFmpeg, ffprobe and encoder options from the local
runtime. Do not switch encoders to hide a failure. Its optional `--foreground`
accepts an aligned straight RGB sequence, e.g. CorridorKey's FG. Otherwise it
uses source RGB plus the mask. It copies audio streams; incompatible MP4/audio
combinations fail without silently transcoding. It needs aligned CFR frames,
not unaligned variable-frame-rate or retimed plate/matte pairs.

The helpers need full FFmpeg with rawvideo input and h264_metadata. On Windows
the pinned imageio-ffmpeg binary is used; bootstrap finds ffprobe or downloads
it locally from a checksum-verified Gyan essentials package. Linux needs its
system FFmpeg/ffprobe package first. Drivers are outside bootstrap's scope.

The compositor expects sRGB RGB (or a known linear foreground), not untreated
log/HDR. It mixes in linear light and tags the limited-range BT.709 YUV output
with sRGB transfer. Inspect colors with a suitable decoder: a tested OpenCV
VideoCapture path interpreted BT.709 output as BT.601 while FFmpeg decoded the
intended RGB values. Don't change the intended transform to match that decoder.

## Isolate a single keying job

`python run.py keyer run-inference ...` scans every ready shot in
`ClipsForInference`. For one requested source use the existing API with the
configured `.venv` interpreter. Bind `root`, `job`, `plate`, `hint` and `runtime`
to inspected absolute paths and the loaded local runtime configuration:

```python
import os
import sys
sys.path.insert(0, str(root))
os.chdir(root)
from clip_manager import ClipAsset, ClipEntry, InferenceSettings, run_inference

clip = ClipEntry("shot", str(job / "shot"))  # Create a fresh shot directory first.
clip.input_asset = ClipAsset(str(plate), "video")
clip.alpha_asset = ClipAsset(str(hint), "sequence")  # Or "video" for a hint movie.
clip.validate_pair()
settings = InferenceSettings(
    image_size=1024, input_is_linear=False, despill_strength=0.5,
    auto_despeckle=False, refiner_scale=1.0, generate_comp=True,
    gpu_post_processing=False, tiled_inference=False, screen_color="green",
)
run_inference([clip], device=runtime["device"], backend="torch",
              settings=settings, max_frames=24)
```

For CPU jobs set `CORRIDORKEY_SKIP_COMPILE=1`. The hint must cover the original
clip; max_frames limits processing, not pairing validation. Remove the limit
for the full export. Blue weights download on demand for blue screen_color;
that checkpoint was not in the initial green-checkpoint hardware smoke test.

Output under the shot's `Output` directory:
- `Matte`: raw linear alpha EXR.
- `FG`: straight foreground RGB in sRGB transfer/gamut.
- `Processed`: linear premultiplied RGBA EXR; use matching alpha interpretation
  and color transforms when importing into Resolve.
- `Comp`: opaque checkerboard PNGs. Make the requested 480p MP4 review explicitly;
  the API's optional automatic comp stitching is not that review step.

## Evidence and limits

See `docs/gpu-validation-2026-11-12.json` for the original synthetic P600/helper
evidence. The P600 reported 4 GiB, driver 582.78, capability 6.1. At internal
512/1024/2048, short tests averaged about 0.97/3.26/34.30 s/frame. The 2048 run
reserved more than dedicated VRAM; prefer 1024 on that card. These historical
observations are not performance guarantees. Ordinary-background segmentation,
B580/QSV execution, production footage and long-video stability were not
established by these checks.
