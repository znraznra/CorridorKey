# CorridorKey: one repository, per-device setup

This checkout contains the adapted engine, the Codex skill under
`.agents/skills/corridorkey`, hint/compositing helpers, compute profiles and setup.
One clone brings all source files. Packages, weights and machine settings are
installed locally and stay outside Git.

## Upload this folder to your GitHub

Create an **empty** GitHub repository (no generated README/licence/gitignore).
Choose its visibility. Keep the upstream CorridorKey name and licence in the
source; its additional terms still apply.

From this checkout, review and publish the local changes:

```powershell
cd D:\Projects\REA\CorridorKey
git status --short
if ((git rev-parse --is-shallow-repository) -eq "true") { git fetch --unshallow origin }
git add .
git diff --cached --stat
git commit -m "Add portable CorridorKey skill and compute profiles"
git remote rename origin upstream
git remote add origin https://github.com/YOUR-USER/YOUR-REPO.git
git push -u origin HEAD:main
```

Replace the example URL. The current origin points to the original author;
renaming it keeps upstream history and avoids pushing to that URL. If you have
already renamed the remotes, inspect `git remote -v` and adapt those commands
instead of running them twice. No force push is needed. Local preparation does
not create a GitHub repo, commit or push.

`.gitignore` excludes `.venv`, `.local`, checkpoints, source clips, outputs,
ordinary `.env` secrets and caches. Inspect staged files before pushing. Do not
use `git add -f` for media, machine config or downloaded binaries.

## Use a new device

Prerequisites: Git, Python 3.12 **or** uv, suitable drivers for GPU use, internet
for the first dependency/model downloads, and sufficient local disk. Linux also
needs FFmpeg including ffprobe (on Arch install `ffmpeg` with the normal package
manager). Bootstrap does not install drivers or replace system Python.
These profiles target Windows/Linux on the supported Torch wheel architectures;
macOS setup has not been provisioned or validated here.

```text
git clone https://github.com/YOUR-USER/YOUR-REPO.git CorridorKey
cd CorridorKey
```

Choose the device profile:

| Device | Setup |
| --- | --- |
| Quadro P600 / Pascal | `python bootstrap.py --profile p600` |
| Intel Arc B580 | `python bootstrap.py --profile arc` |
| Compatible newer NVIDIA GPU | `python bootstrap.py --profile cuda` |
| CPU | `python bootstrap.py --profile cpu` |

On Linux use `python3` if needed. On Windows `py bootstrap.py ...` also works.
Bootstrap itself can start under another Python version; it locates Python 3.12
or asks uv to provide it for `.venv`.

Setup installs the selected Torch wheels, EXR-compatible OpenCV and media
helpers, runs real tensor operations and downloads the official green checkpoint
(about 399 MB). Blue weights download on demand. Windows bootstrap locates
ffprobe or downloads it to `.local/tools` from a checksum-verified Gyan package;
the pinned FFmpeg encoder comes from imageio-ffmpeg. Use `--ffprobe /path/to/ffprobe`
to select an existing probe executable. `--dry-run` prints the plan without
installing. `--skip-model` defers downloading green weights to first use.

Open the cloned folder in Codex and invoke **$corridorkey**. The skill is
repo-scoped; no second skill clone or hardcoded personal path is needed.
See the official discovery documentation:
https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills

## Local commands

```text
python run.py check
python run.py hint --video /path/to/Input.mp4 --output /path/to/job/AlphaHint --screen green
python run.py green --video /path/to/Input.mp4 --matte /path/to/job/Matte --output /path/to/job/preview.mp4 --height 480 --max-frames 24 --bitrate 4M
python run.py keyer run-inference --help
```

Run from the clone, or invoke run.py by absolute path with absolute media paths.
The launcher resolves interpreter/tool paths from ignored `.local/runtime.json`.
Each machine creates that file; do not copy it between devices.

`hint` makes a coarse hint for existing screen footage. `green` composites using
a supplied matte; it does no ordinary-background segmentation. For one keying
job the skill uses the isolated API recipe in its runtime reference. The
upstream `keyer run-inference` command scans every ready shot, so use it
intentionally for batches.

Profile encoders are candidates, not hardware/performance promises. The P600
was tested; B580/Arch needs its first on-device check. Modern CUDA and CPU
profiles don't imply equivalent speed. Optional AI hint generators and
long-video stability remain unvalidated.

## Update later

```text
git pull --ff-only
python bootstrap.py --profile YOUR-PROFILE
```

See README_GPU.md and docs/gpu-validation-2026-11-12.json for the original
synthetic measurements and their limits.
