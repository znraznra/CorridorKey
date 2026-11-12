"""Make aligned grayscale PNG hints for footage already shot on green/blue."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def chroma_alpha(bgr, screen, low=0.05, high=0.3, erosion=1, blur=3):
    rgb = bgr[..., ::-1].astype(np.float32) / 255
    channel = 1 if screen == "green" else 2
    primary = rgb[..., channel]
    others = np.max(np.delete(rgb, channel, axis=2), axis=2)
    dominance = (primary - others) / np.maximum(primary, 0.001)
    alpha = 1 - np.clip((dominance - low) / (high - low), 0, 1)
    if erosion:
        alpha = cv2.erode(alpha, np.ones((erosion * 2 + 1, erosion * 2 + 1), np.uint8))
    if blur > 1:
        alpha = cv2.GaussianBlur(alpha, (blur, blur), 0)
    return np.round(alpha * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screen", choices=("green", "blue"), required=True)
    parser.add_argument("--low", type=float, default=0.05)
    parser.add_argument("--high", type=float, default=0.3)
    parser.add_argument("--erosion", type=int, default=1)
    parser.add_argument("--blur", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.low < args.high <= 1 or args.erosion < 0 or args.blur < 1 or args.blur % 2 != 1:
        parser.error("Use 0 <= low < high <= 1, erosion >= 0, and positive odd blur.")
    if not args.video.is_file():
        parser.error("Input video does not exist.")
    if args.output.exists():
        parser.error("Use a fresh output directory; existing hints are never overwritten.")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError("Cannot decode input video.")
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    args.output.mkdir(parents=True)
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            alpha = chroma_alpha(frame, args.screen, args.low, args.high, args.erosion, args.blur)
            if not cv2.imwrite(str(args.output / f"{count:06d}.png"), alpha):
                raise RuntimeError("Could not write mask frame.")
            count += 1
    finally:
        cap.release()
    if not count or (expected > 0 and count != expected):
        raise RuntimeError(f"Decoded {count} frames; expected {expected}. Do not use this incomplete hint.")
    report = {
        "source": str(args.video.resolve()),
        "frames": count,
        "source_fps": fps,
        "screen": args.screen,
        "low": args.low,
        "high": args.high,
        "erosion": args.erosion,
        "blur": args.blur,
        "method": "coarse chroma dominance; no AI segmentation",
        "status": "complete",
    }
    (args.output / "hint-metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
