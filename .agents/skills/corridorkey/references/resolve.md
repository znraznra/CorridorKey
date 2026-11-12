# Resolve coarse alpha hint

Use when the user wants to prepare or refine a hint in DaVinci Resolve. Work on
a duplicate/test timeline. Preserve the same source range, dimensions and fps
for the untouched plate and hint exports.

1. In Fusion, insert Delta Keyer after MediaIn and sample the screen with its
   Background Color eyedropper. Inspect alpha and refine the Matte controls for
   a roughly solid subject and empty background. Soft, slightly contracted edges
   are suitable hints; do not spend time matching final hair detail unnecessarily.
2. Add Channel Booleans after the keyer. Connect the keyer to its yellow
   Background input. Add a Background node with Alpha=1 to its green Foreground
   input. Set Operation=Copy, To Red/Green/Blue=Alpha BG, To Alpha=Alpha FG.
3. Connect Channel Booleans to MediaOut. In RGB view the result must be a visible
   grayscale subject on black with opaque alpha. Viewer-only Alpha display does
   not change what RGB will be exported.
4. Export this as a regular grayscale MP4 `AlphaHint.mp4` or a lossless PNG
   sequence. Match the plate's exact frame count/start/fps. No transparent codec
   is needed for a visible grayscale hint. Keep test stills outside the sequence.

For ordinary backgrounds the subject must first be isolated using available
masking/tracking/roto tools; Delta Keyer isn't a general-purpose segmenter. Check
the user's Resolve edition/version before promising a particular AI tool.

Sources checked 2026-11-12:
- https://documents.blackmagicdesign.com/UserManuals/DaVinci-Resolve-18-Fusion-Visual-Effects.pdf
- https://documents.blackmagicdesign.com/dk/UserManuals/Fusion7_Tool_Reference.pdf

If exact controls differ, consult the manual for the installed Resolve version.
