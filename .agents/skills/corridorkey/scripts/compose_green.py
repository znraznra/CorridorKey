"""Composite aligned foreground/matte frames onto green; performs no segmentation.

Requires CFR source video and matching source-resolution masks. Copies source
audio into MP4. Encoder selection is explicit; no automatic encoder fallback.
"""

# OpenEXR must be enabled before importing OpenCV.
# ruff: noqa: E402
import argparse
import json
import os
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np


def sequence(path):
    return sorted(
        [p for p in path.iterdir() if p.suffix.lower() in (".png", ".exr", ".tif", ".tiff", ".jpg")],
        key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.name)],
    )


def normalized(image):
    if image is None:
        raise RuntimeError("A required image could not be decoded.")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535
    else:
        image = image.astype(np.float32)
    if not np.isfinite(image).all():
        raise RuntimeError("Non-finite image values.")
    return image


def srgb_to_linear(value):
    return np.where(value <= 0.04045, value / 12.92, ((np.maximum(value, 0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * np.maximum(value, 0) ** (1 / 2.4) - 0.055)


def probe(binary, path, count=False):
    cmd = [str(binary), "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if count:
        cmd.append("-count_frames")
    return json.loads(subprocess.check_output([*cmd, str(path)], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video", "matte", "output", "ffmpeg"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--foreground", type=Path)
    parser.add_argument("--fg-space", choices=("srgb", "linear"), default="srgb")
    parser.add_argument("--encoder", choices=("h264_nvenc", "h264_qsv", "libx264"), required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--bitrate", default="8M")
    parser.add_argument("--height", type=int)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".mp4":
        parser.error("This helper delivers MP4; choose another workflow for alpha-capable masters.")
    if not args.video.is_file() or not args.matte.exists() or not args.ffmpeg.is_file():
        parser.error("Missing source, matte, or FFmpeg binary.")
    if args.foreground and not args.foreground.is_dir():
        parser.error("--foreground must be a straight RGB image sequence directory.")
    if args.height is not None and (args.height < 2 or args.height % 2):
        parser.error("--height must be positive and even.")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive.")
    if not re.fullmatch(r"[1-9]\d*(?:[kKmM])?", args.bitrate):
        parser.error("Use a positive bitrate, e.g. 8M.")
    if args.gpu_index < 0:
        parser.error("--gpu-index cannot be negative.")
    probe_bin = args.ffprobe or args.ffmpeg.with_name("ffprobe" + args.ffmpeg.suffix)
    if not probe_bin.is_file():
        found = shutil.which("ffprobe")
        if not found:
            parser.error("Supply --ffprobe or place it beside FFmpeg.")
        probe_bin = Path(found)
    log_path = args.output.with_suffix(".mp4.ffmpeg.log")
    report_path = args.output.with_suffix(".mp4.json")
    if any(p.exists() for p in (args.output, log_path, report_path)):
        parser.error("Choose a fresh output filename; outputs/logs are never overwritten.")
    source_info = probe(probe_bin, args.video)
    video = next(s for s in source_info["streams"] if s["codec_type"] == "video")
    fps = Fraction(video["avg_frame_rate"])
    if fps <= 0 or fps != Fraction(video["r_frame_rate"]):
        parser.error("A CFR source is required. Normalize plate and matte together before running.")
    if any(int(side.get("rotation", 0)) % 360 for side in video.get("side_data_list", [])):
        parser.error("Bake source rotation into aligned plate and matte pixels first.")
    width, height = int(video["width"]), int(video["height"])
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError("Source decoder could not open video.")
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if expected < 1:
        cap.release()
        raise RuntimeError("Could not establish source frame count.")
    count = min(expected, args.max_frames) if args.max_frames else expected
    matte_files = sequence(args.matte) if args.matte.is_dir() else None
    fg_files = sequence(args.foreground) if args.foreground else None
    matte_cap = cv2.VideoCapture(str(args.matte)) if matte_files is None else None
    try:
        mask_count = len(matte_files) if matte_files is not None else int(matte_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for label, available in (
            ("matte", mask_count),
            ("foreground", len(fg_files) if fg_files is not None else expected),
        ):
            if available < count or (not args.max_frames and available != expected):
                raise RuntimeError(f"{label} frame count {available} does not match the requested {count} frames.")
        out_h = args.height or height
        out_w = round(width * out_h / height / 2) * 2 if args.height else width
        if out_w < 2 or out_w % 2 or out_h % 2:
            raise RuntimeError("YUV420 H.264 requires even dimensions; choose an explicit approved resize.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(args.ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{out_w}x{out_h}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-i",
            str(args.video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            args.encoder,
            "-b:v",
            args.bitrate,
        ]
        if args.encoder == "h264_nvenc":
            cmd += ["-gpu", str(args.gpu_index)]
        cmd += [
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "iec61966-2-1",
            "-bsf:v",
            "h264_metadata=colour_primaries=1:transfer_characteristics=13:matrix_coefficients=1:video_full_range_flag=0",
            "-c:a",
            "copy",
            "-t",
            f"{float(count / fps):.12f}",
            "-movflags",
            "+faststart",
            str(args.output),
        ]
        with log_path.open("wb") as log:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            try:
                for i in range(count):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Source decoding stopped at frame {i}.")
                    if matte_files is not None:
                        matte = cv2.imread(str(matte_files[i]), cv2.IMREAD_UNCHANGED)
                    else:
                        ok, matte = matte_cap.read()
                        if not ok:
                            raise RuntimeError(f"Matte decoding stopped at frame {i}.")
                    alpha = normalized(matte)
                    if alpha.ndim == 3:
                        alpha = alpha[..., 0]
                    if alpha.shape != (height, width):
                        raise RuntimeError("Matte dimensions must match source; no implicit alignment resize.")
                    foreground = cv2.imread(str(fg_files[i]), cv2.IMREAD_UNCHANGED) if fg_files is not None else frame
                    foreground = normalized(foreground)
                    if foreground.shape != (height, width, 3):
                        raise RuntimeError("Foreground must be aligned straight RGB at source dimensions.")
                    foreground = foreground[..., ::-1]
                    fg_linear = srgb_to_linear(foreground) if args.fg_space == "srgb" else foreground
                    alpha = np.clip(alpha[..., None], 0, 1)
                    composite = fg_linear * alpha + np.array([0, 1, 0], np.float32) * (1 - alpha)
                    rgb = np.clip(linear_to_srgb(composite), 0, 1)
                    if (out_h, out_w) != (height, width):
                        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    process.stdin.write(np.round(rgb * 255).astype(np.uint8).tobytes())
            except BaseException:
                if process.poll() is None:
                    process.kill()
                process.wait()
                raise
            finally:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            if process.wait() != 0:
                raise RuntimeError(f"FFmpeg failed; inspect {log_path}. No fallback encoder was used.")
    finally:
        cap.release()
        if matte_cap is not None:
            matte_cap.release()
    output_info = probe(probe_bin, args.output, count=True)
    rendered = next(s for s in output_info["streams"] if s["codec_type"] == "video")
    if int(rendered["nb_read_frames"]) != count or Fraction(rendered["avg_frame_rate"]) != fps:
        raise RuntimeError("Encoded frame count/fps verification failed.")
    if (int(rendered["width"]), int(rendered["height"])) != (out_w, out_h):
        raise RuntimeError("Encoded dimensions verification failed.")
    for key, expected_value in (
        ("color_space", "bt709"),
        ("color_transfer", "iec61966-2-1"),
        ("color_primaries", "bt709"),
        ("color_range", "tv"),
    ):
        if rendered.get(key) != expected_value:
            raise RuntimeError(f"Encoded color metadata verification failed: {key}.")
    source_audio = [s["codec_name"] for s in source_info["streams"] if s["codec_type"] == "audio"]
    output_audio = [s["codec_name"] for s in output_info["streams"] if s["codec_type"] == "audio"]
    if source_audio != output_audio:
        raise RuntimeError("Audio stream count/codec verification failed.")
    report = {
        "source": str(args.video.resolve()),
        "matte": str(args.matte.resolve()),
        "output": str(args.output.resolve()),
        "frames": count,
        "fps": str(fps),
        "dimensions": [out_w, out_h],
        "encoder_requested": args.encoder,
        "audio_codecs_copied": output_audio,
        "foreground_space": args.fg_space,
        "encoded_transfer": "sRGB",
        "background_rgb": [0, 255, 0],
        "log": str(log_path.resolve()),
        "status": "encoded and probed; inspect edges/audio perceptually",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
