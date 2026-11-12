---
name: corridorkey
description: "Key green/blue-screen video with CorridorKey, create alpha hints, or put a masked subject onto green. Ordinary backgrounds need separate matting."
---

# CorridorKey video

Use the adapted CorridorKey engine shipped in this repository. Both directions
are supported as workflows, but their prerequisites differ. Resolve guidance is
available when that is the user's editor.

## Choose the direction

- **Remove an existing green/blue screen:** run CorridorKey with the original
  plate plus an aligned coarse alpha hint. Deliver a transparent image sequence
  or another explicitly requested alpha-capable format.
- **Put a subject onto green:** first obtain an aligned subject matte, then
  composite the foreground onto solid green. Ordinary-background segmentation
  is a separate operation; CorridorKey is not an arbitrary-background remover.
- Clarify genuinely ambiguous "green screen conversion" requests. Reuse a
  direction already established in the chat. Do not ask again when it is clear.

Read [the runtime reference](references/runtime-portable.md) before executing a job. It
identifies the installed checkout, interpreter, verified hardware settings,
commands and evidence. Locate this checkout and its ignored runtime configuration;
no particular drive or user directory is required. The B580/Arch path needs
hardware validation.

An uploaded skill bundle contains instructions/helpers, not the engine,
environment or a connection to the user's GPU. Before executing, establish
actual access to the checkout and its local runtime. If the harness only has
cloud execution or read-only file access, explain that limitation and provide
local commands or hand off to a local coding session. Do not claim a desktop
GPU was used from a cloud sandbox.

## Prepare the job

Inspect the supplied video and representative frames. Establish the subject,
desired direction, output format and timing only where these are not already
specified. Preserve source files. Use a separate local job folder and process
only this job, not every ready clip in the checkout. Probe dimensions, rational
frame rate, duration, frame count, audio and rotation. Check storage before
large EXR/PNG sequences. Variable-frame-rate or retimed footage needs explicit
alignment of the plate and matte; the helpers below assume aligned CFR media.

### Existing screen footage

Use a supplied hint if valid. Otherwise `scripts/make_chroma_hint.py` can generate
a lightweight coarse hint with ordinary chroma math on CPU. This is not an AI
segmenter. Inspect early/middle/late masks, especially green clothing, shadows,
motion blur and occlusions; tune its thresholds or use a better supplied matte
if the subject is cut away or background is retained. Keep preview images
outside the hint sequence directory because the keyer counts every image there.

For a Resolve-produced hint, read [the Resolve recipe](references/resolve.md).
The hint must be visible grayscale RGB: white subject, black background, soft
gray edges. Viewing alpha in a viewer alone does not change exported RGB.
Preserve frame alignment and count. Avoid large dilated foreground halos.

Run a short CorridorKey sample before the full clip. Use the backend factory so
the P600 receives FP32 eager execution. Prefer internal 1024 for this P600 and
512 for drafts; internal size is distinct from delivery resolution. Do not
silently lower requested processing quality. Keep CPU post-processing unless
the selected GPU path has been checked. Preserve alpha as linear data and the
documented foreground/premultiplication conventions in the runtime reference.

### Ordinary footage to green

Use an existing tracked/AI/roto subject matte where available. If none exists,
choose and validate a separate local matting method on a short sample, or guide
the user through Resolve masks/tracking. The local optional BiRefNet/GVM/
VideoMaMa generators have not been validated on the P600 or B580. Do not claim
they are installed, fast, temporally stable or compatible merely because their
wrapper code exists. Do not route ordinary footage through CorridorKey with a
white hint and describe the result as background removal.

With a usable matte, use `scripts/compose_green.py` for local compositing. For
screen footage first keyed through CorridorKey, its straight `FG` and refined
`Matte` sequences can be passed to this helper. For ordinary footage it can use
the source RGB plus the supplied matte; inspect color contamination/halos at
mixed edges since that does not reconstruct true foreground color. Use a true
foreground estimate if the available matting method provides one.

## Review and deliver

Use the requested review format; a **local 480p MP4 preview** is the included
workflow's default, with green or a checkerboard for transparent output. Follow
the current user's instructions for previews and full-resolution delivery.
Retain source dimensions, frame rate, timing
and audio unless the user requested a change. Plain H.264 MP4 does not carry the
transparent master; use EXR/PNG alpha or an agreed alpha-capable codec for that.

For `compose_green.py`, choose the encoder explicitly. On the tested P600 use
`h264_nvenc` with explicit bitrate; inspect the encoder log and output. No
automatic encoder fallback is implemented. Observe applicable host/Remotion
GPU rules when those tools are actually used; these helpers are not Remotion.

Verify decoded frames, frame count/duration, joins if any, audio continuity and
synchronization, and edge quality across representative motion. Distinguish
synthetic checks from real-footage approval and short samples from full-video
stability. Link local preview/master files and state relevant limits. Keep
outputs in the selected job destination. This repository does not define
workstation or company policy and does not authorize uploads or publication.
