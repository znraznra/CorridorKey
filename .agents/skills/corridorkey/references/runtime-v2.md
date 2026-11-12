# CorridorKey 2 / Intel Arc

Locate the checkout as in runtime-portable.md, then read `docs/CORRIDORKEY2.md`
for installation, limits, commands and licence terms. Version 2 is opt-in:
`python bootstrap.py --engine v2 --profile arc`. The P600 stays on version 1.
Use `.local/runtime-v2.json` and `.local/ck2/venv`; do not install v2 packages
into `.venv`, change the default engine or copy another machine's settings.

Use `run.py --engine v2 check`, then `run.py --engine v2 verify --output
<fresh-job-directory>` on the B580 before treating the pipeline as usable.
The development machine has no B580, so mocks, CPU checks and clean patch
application do not establish XPU inference support. Inspect the reported GPU
name/index and the separate validation report. Never fall back to CPU/CUDA or
another engine after a failed explicit request.

For a single source, `run.py --engine v2 keyer --input <absolute-plate>
--output_dir <fresh-absolute-output> --num_frames 12 --outputs alpha fg cutout`
uses the actual CK2 CLI. The default output format is EXR. Hints are optional:
`--hint_first_frame <grayscale-image>` or `--hint_video <aligned-video-or-dir>`.
Use the existing Resolve reference for visible grayscale hint preparation.
Validate source/hint alignment and inspect edges/motion on real footage.

V2 defaults to a small staged temporal window because its CUDA VRAM estimates
are not validated on Intel. Native delivery dimensions are preserved; a short
temporal window may affect consistency. Do not silently change processing
settings or source color interpretation merely to finish a job.

V2 alpha is linear scalar data. FG EXR is straight scene-linear RGB; cutout EXR
is straight scene-linear RGBA with upstream despill. V1's premultiplied
Processed convention does not apply. For the shared green helper, supply the
CK2 alpha and FG directories, `--fg-space linear`, and explicit job
paths. Check this helper's current options before use. EXRs and upstream video
previews contain no audio: retain the source plate for an explicit audio
mux/export, and never claim audio was copied by the keyer.

Encoded opaque CK2 previews select QSV with an explicit bitrate. QSV can select
an Intel iGPU independently of the PyTorch device, so verify media-engine
selection separately. Transparent cutout video is rejected; use EXR. Follow
the current user's local review and delivery requirements. Ordinary-background
matting remains a separately assessed workflow; do not promise universal
background removal from the existence of no-hint mode.
