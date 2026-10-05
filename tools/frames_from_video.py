"""Turn a short video into training frames: one clip per class (or several), frames in data/raw/<class>/.

    python -m tools.frames_from_video cow_pasture.mp4 --label cow
    python -m tools.frames_from_video horse_*.mov --label horse --every 0.25
    python -m tools.frames_from_video clip.mp4 --label cow --max 40 --side 480

Every frame is named <video>_<number>.jpg, e.g. cow-pasture_0012.jpg. The part before the last "_" says
which video a frame came from, so stage 1 can keep all frames of one video on the same side of the
train/test split (group_by: "prefix" in 01_image-processing/digital-data_prepare/action.yaml).
Neighbouring frames are near-copies: split them apart and the test set leaks into training.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def extract(video: Path, label: str, out: Path, every: float, side: int, max_frames: int | None) -> int:
    if not video.is_file():
        raise SystemExit(f"❌ There's no video at {video}. Check the path.")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"❌ OpenCV can't open {video}. An iPhone .mov in HEVC? Export it as 'Most Compatible' "
                         f"(H.264), or convert it: ffmpeg -i {video.name} {video.stem}-h264.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(every * fps))                       # keep one frame every `every` seconds
    stem = re.sub(r"[^A-Za-z0-9-]+", "-", video.stem).strip("-") or "video"
    folder = out / label
    folder.mkdir(parents=True, exist_ok=True)
    n = kept = 0
    while max_frames is None or kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if n % step == 0:
            h, w = frame.shape[:2]
            if side and max(h, w) > side:                   # small files: the pipeline shrinks to 128 px anyway
                s = side / max(h, w)
                frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(folder / f"{stem}_{kept:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            kept += 1
        n += 1
    cap.release()
    print(f"🎬 {video.name}: {n} frames at {fps:.0f} fps → kept {kept} (one every {step}) in {folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder}/{stem}_*.jpg")
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path, help="one or more video files of the same class")
    ap.add_argument("--label", required=True, help="the class, e.g. cow: becomes the folder data/raw/<label>/")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--every", type=float, default=0.5, help="seconds between kept frames (default 0.5)")
    ap.add_argument("--side", type=int, default=640, help="shrink the longest side to this many pixels (0 = keep)")
    ap.add_argument("--max", type=int, default=None, help="at most this many frames per video")
    a = ap.parse_args()
    total = sum(extract(v, a.label, a.out, a.every, a.side, a.max) for v in a.videos)
    print(f"✅ {total} frames for '{a.label}'. Next: source: \"folder\" and group_by: \"prefix\" in "
          f"01_image-processing/digital-data_prepare/action.yaml")
