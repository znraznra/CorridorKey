# CorridorKey — portable video keying and skills

A local adaptation of [CorridorKey by Corridor Digital](https://github.com/nikopueringer/CorridorKey), with a tested Quadro P600 path, an Intel Arc profile, and a reusable video workflow for Codex, Claude Code, Claude Desktop and other tool-enabled harnesses.

**One clone contains the engine, skill and helpers. Each device still needs bootstrap, compatible drivers and its own local environment.** This repository provides a CLI and skills, not a graphical editor or a local-GPU bridge for cloud chat.

## Current changes

- Pascal-compatible PyTorch/CUDA 12.6 setup and FP32 eager inference for the P600.
- Intel XPU device selection, indexed devices, cache handling and an Arc setup profile.
- Explicit modern NVIDIA and CPU profiles; no automatic fallback from a failed GPU request.
- A repo-local `$corridorkey` skill, a Claude Code adapter and a skill-ZIP packager.
- Per-device runtime discovery instead of hardcoded user, drive or Remotion-project paths.
- A coarse green/blue chroma-hint generator and a matte-based green-background compositor.
- Explicit encoder selection, frame/color checks, source preservation and audio stream copying in the compositor.
- OpenCV 4.12.0.88 pinned for EXR writing; media, weights, tools and environments excluded from Git.

## What it can do

| Request | Workflow | Current limit |
| --- | --- | --- |
| Remove an existing green/blue screen | Original plate + aligned coarse hint -> CorridorKey | Green checkpoint tested; blue path provided but not in the original hardware test |
| Create an alpha mask | Generate a coarse hint, then produce the refined `Matte` EXR pass | Automatic hint helper is ordinary chroma math, not general AI segmentation |
| Put an isolated subject onto green | Straight foreground/source RGB + aligned matte -> green MP4 | Ordinary backgrounds first need a separate tracked/AI/roto matte |
| Preserve transparency | `Processed` RGBA EXR sequence, plus separate FG/Matte passes | Plain H.264 MP4 has no alpha; other transparent formats need an explicit export step |

A white hint does not turn CorridorKey into an arbitrary-background remover. Optional BiRefNet, GVM and VideoMaMa wrappers exist upstream, but their execution and memory requirements have not been validated for these profiles.

## Levels of support

| Component | Status | Evidence / boundary |
| --- | --- | --- |
| Quadro P600, Windows | Locally tested, short synthetic clips | Trained model at 512/1024/2048, normal batch launcher, EXR readback and GPU operations passed |
| Intel Arc B580, Arch Linux | Experimental; on-device validation pending | XPU routing/error handling unit-tested; no B580 execution or QSV throughput claim |
| Newer NVIDIA / cu128 | Setup profile provided, not hardware-validated here | Select only for a GPU supported by that wheel build; not Pascal |
| CPU | Explicit profile provided | CPU reference inference was checked in the original test; fresh CPU-wheel bootstrap/performance not established |
| Hint/compositing helpers | Short synthetic tests passed | Green/blue hint direction, frame alignment, NVENC, full A/V decode and copied AAC identity checked |
| Real production footage / long videos | Not validated by the synthetic baseline | Inspect edges, motion, memory and audio on the actual job |
| macOS | Not provisioned by this bootstrap | The current setup profiles target Windows/Linux |
| GUI | Not included | CLI plus terminal wizard; separate GUI forks are not integrated here |

The original test suite passed 438 tests with one optional MLX skip before the later portability additions. Additional portability checks have their own test file; the historical count is not a claim that every later revision or hardware path was tested.

Historical P600 measurements on 1280x720 synthetic inputs, FP32 eager with CPU post-processing:

| Internal model size | Mean seconds/frame | Peak allocated memory |
| --- | ---: | ---: |
| 512 | 0.97 | 0.50 GiB |
| 1024 | 3.26 | 1.17 GiB |
| 2048 | 34.30 | 3.86 GiB |

The 2048 run reserved 6.35 GiB, above the reported 4 GiB dedicated VRAM. Prefer 1024 on this P600; a short completion does not prove a reliable VRAM fit. Internal model size is separate from output dimensions. These are observations, not speed guarantees.

See [recorded validation](docs/gpu-validation-2026-11-12.json) and [initial GPU guide](README_GPU.md).

## Clone and set up

Prerequisites: Git; Python 3.12 or [uv](https://docs.astral.sh/uv/getting-started/installation/); sufficient local disk; internet for first dependency/model downloads; and the appropriate GPU drivers. Linux also needs FFmpeg including ffprobe (`ffmpeg` in Arch's package manager). Arc needs suitable Intel compute/Level Zero drivers. Setup does not install drivers or replace system Python.

```text
git clone https://github.com/YOUR-USER/YOUR-REPO.git CorridorKey
cd CorridorKey
```

Replace the URL after publishing your repository. Then choose **one** profile:

| Machine | Command |
| --- | --- |
| P600 / Pascal | `python bootstrap.py --profile p600` |
| Arc B580 | `python bootstrap.py --profile arc` |
| Compatible newer NVIDIA | `python bootstrap.py --profile cuda` |
| CPU | `python bootstrap.py --profile cpu` |

Use `python3` on Linux or `py` on Windows if that is your launcher name. Bootstrap can start under another Python version and locates Python 3.12 or uv for the actual environment.

Bootstrap installs packages in `.venv`, checks real tensor execution, locates media tools and writes ignored `.local/runtime.json`. It downloads the official green checkpoint (about 399 MB); blue weights download when selected. Windows can obtain a missing ffprobe from a published checksum-verified Gyan package. Model weights and tools are not committed.

Options: `--dry-run` prints the plan; `--with-tests` adds test tools; `--skip-model` defers weights; `--ffprobe PATH` selects an existing probe binary. Do not maintain these profiles with upstream `uv sync --extra cuda`: it can restore a P600-incompatible cu128 wheel.

```text
python run.py check
python run.py --help
```

`check` validates compute, not video encoder hardware. Run a representative short export on each new device. Rerun bootstrap after moving to a new device or updating runtime dependencies. Never copy `.local/runtime.json` between machines.

The portable bootstrap rerun passed on the existing P600 installation. A clean-machine end-to-end bootstrap and every supported app's UI were not exercised here.

## Harness integration status

| Harness | Included linkage | Validation level |
| --- | --- | --- |
| Codex | Repo-local `.agents/skills/corridorkey` | Skill format, helpers and local CLI validated; discovery follows current documentation |
| Claude Code | Thin `.claude/skills/corridorkey` adapter and CLAUDE.md | Adapter format validated; Claude Code UI not exercised |
| Claude Desktop Code/local session | Local checkout + Claude Code adapter | Requires its actual local execution tools; not UI-tested here |
| Claude Desktop chat/Cowork | Uploadable skill ZIP | Instructions only until authorized local execution is available; GPU linkage not supplied |
| Other harness/CLI | Canonical SKILL.md + Python entry points | Requires compatible file/shell tools; app-specific registration is not automatically guaranteed |

## Use with Codex

Open this cloned folder as the Codex project. The maintained skill is [.agents/skills/corridorkey/SKILL.md](.agents/skills/corridorkey/SKILL.md). Repo-local discovery and `$skill` invocation follow the [official Codex skill documentation](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills).

Example prompt:

```text
Use $corridorkey to remove the green screen from <absolute video path>.
Use the bootstrapped profile, create a coarse hint if needed, and process a
short sample first. Give me a local 480p MP4 checkerboard preview and the
RGBA EXR output. Preserve source timing and audio; report any limitations.
```

If it does not appear, check the project folder and `/skills`; Codex documents automatic skill detection and restarting if an update does not appear. No separate skill clone is required.

## Use with Claude Code

The checked-in [.claude/skills/corridorkey/SKILL.md](.claude/skills/corridorkey/SKILL.md) is a thin adapter pointing to the canonical workflow. [CLAUDE.md](CLAUDE.md) supplies project navigation. Start Claude Code from the cloned folder, with local shell/file access, then use:

```text
/corridorkey Put the subject in <absolute video path> onto a solid green
background. Inspect whether a subject matte already exists. If the source
has an ordinary background, establish a separate matting step first.
Use the local runtime, preserve timing/audio and make a 480p preview first.
```

The `.claude/skills/<name>/SKILL.md` layout and slash-command invocation are documented by [Claude Code](https://code.claude.com/docs/en/skills). The adapter files are provided and inspectable; Claude Code UI execution was not tested in this session.

## Use with Claude Desktop

Desktop has multiple execution surfaces. A Code/local coding session can work with the cloned folder and Claude Code adapter. An uploaded chat/Cowork skill carries instructions; **uploading it does not give the chat access to this checkout, Python environment or GPU**.

Build the bundle locally:

```text
python scripts/package_skill.py
```

This creates ignored `.local/dist/corridorkey.zip`, containing the skill, helpers, references and licence, without engine weights or environments. In Claude's current skill UI, enable code execution/file creation as needed, then use Customize -> Skills -> Create skill -> Upload a skill and enable the bundle. Account/workspace controls can affect availability. See [Claude's upload instructions](https://support.claude.com/en/articles/12512180-use-skills-in-claude) and [custom-skill guidance](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills).

Example Desktop prompt:

```text
Use my CorridorKey skill. The cloned checkout is <absolute repository path>
and the input is <absolute video path>. First verify whether your tools can
read and execute the local runtime and access its selected compute device.
If you only have cloud execution or read-only file access, give me the local
commands or hand off to a local coding session; do not claim local GPU use.
Then create a short 480p review sample for <remove screen / create alpha /
put subject onto green>, preserving timing and audio.
```

This repository includes no MCP server or automatic Desktop-to-GPU connection. A filesystem-only connector is insufficient for execution. Use a local coding session or a separately configured, authorized execution bridge; no particular bridge is installed by these files.

## Other harnesses, CLIs and adapting a skill

Any harness with local file reading and shell/subprocess execution can load the canonical SKILL.md and use `bootstrap.py` / `run.py`. Skill-format support alone does not provide compute access. [Agent Skills' integration guide](https://agentskills.io/client-implementation/adding-skills-support) describes discovery, metadata and loading. If your harness has no skill system, provide the file path in its prompt and ask it to read the relevant references.

Prompt to register/adapt the skill in another harness:

```text
Read <repo>/.agents/skills/corridorkey/SKILL.md and the referenced runtime guide.
Register this workflow using this harness's documented skill mechanism.
Keep the repository skill and helper scripts as the source of truth; create
only a thin adapter if needed. Discover paths from the cloned checkout and
its ignored local runtime config. Do not hardcode a username, drive letter
or another project's tools. State whether you can actually run the local
Python/GPU workflow. Keep ordinary-background matting separate from screen
keying, and preserve the documented support limits. Validate the adapter.
```

For actual work without automatic skill loading:

```text
Read <repo>/.agents/skills/corridorkey/SKILL.md. Use that workflow to create
an alpha mask for <video>. Inspect the background first: generate a chroma
hint only for an existing screen, or establish a separate subject matte.
Use <repo>/run.py and the selected runtime. Start with a short local sample.
```

## Direct local commands

```text
python run.py hint --video /path/to/Input.mp4 --output /path/to/job/AlphaHint --screen green
python run.py green --video /path/to/Input.mp4 --matte /path/to/job/Matte --output /path/to/job/preview.mp4 --height 480 --max-frames 24 --bitrate 4M
python run.py keyer run-inference --help
```

Run from the clone, or use absolute launcher/media paths. The `green` helper accepts `--foreground` for a straight FG sequence and has explicit encoder selection; it never silently substitutes an encoder. Use `--height 480` for review; omit height for source dimensions. Removing `--max-frames` requests the full frame range.

The upstream batch keyer scans every ready shot under `ClipsForInference`. For an isolated video, follow the skill's [single-job API recipe](.agents/skills/corridorkey/references/runtime-portable.md#isolate-a-single-keying-job). Hints must be frame-aligned and cover the original clip. The helpers assume CFR media and matching pixel orientation/dimensions; normalize retimed/VFR media deliberately rather than guessing alignment.

Output passes: `Matte` is linear alpha EXR; `FG` is straight sRGB foreground EXR; `Processed` is linear premultiplied RGBA EXR; `Comp` is an opaque checkerboard PNG sequence. EXR sequences contain no audio. The green compositor copies source audio when MP4-compatible and fails clearly otherwise. For sRGB/linear interpretation and color checks, follow the runtime reference.

For a manually prepared hint in Resolve, use the [Resolve recipe](.agents/skills/corridorkey/references/resolve.md). Viewer-only Alpha display does not change exported RGB: render the matte as visible grayscale.

## Upload and update your repository

Follow [PORTABLE.md](PORTABLE.md#upload-this-folder-to-your-github) for the exact commands, including completing this shallow clone's history if needed, committing your changes, preserving the author remote as `upstream` and pushing to your own empty GitHub repository. No remote was changed or upload performed by local preparation.

Later, `git pull --ff-only` updates source; rerun bootstrap for your selected profile. Keep `.venv`, `.local`, inputs, outputs and checkpoints out of Git. The skill stays in the same clone as the engine.

## Attribution and upstream documentation

Original engine and model: Corridor Digital / Niko Pueringer. Local adaptation and workflow preparation: Arnold Libelo. Retain the CorridorKey name and [original licence with its additional terms](LICENSE). Optional third-party modules retain their own terms.

[Original upstream README](docs/UPSTREAM_README.md) is preserved for reference. Its generic installer and hardware guidance does not override the profile-specific setup above.
