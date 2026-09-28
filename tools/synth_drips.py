"""Synthetic camera frames of an IV drip chamber — the Drip Detector's "day zero" dataset.

A startup rarely has real data on day one, so we fake it until we make it: every
frame is drawn with OpenCV and then *damaged* the way a cheap clip-on camera would
damage it (tilt, dim light, sensor noise, dead pixels, blur, JPEG). Each pipeline
stage exists to undo or exploit one of those effects.

Classes
    drop       a drop is falling through the chamber  → infusion is running
    no_drop    no falling drop                          → infusion stopped / blocked
    low_fluid  chamber nearly empty                     → risk of air in the line

Run standalone:  python -m tools.synth_drips --out data/raw --per-class 150
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("drop", "no_drop", "low_fluid")
GENERATOR_VERSION = 2  # bump when draw_frame changes so cached datasets are rebuilt


def _ellipse(img, center, axes, color, thickness=-1):
    c = (int(round(center[0])), int(round(center[1])))
    a = (max(1, int(round(axes[0]))), max(1, int(round(axes[1]))))
    cv2.ellipse(img, c, a, 0, 0, 360, color, thickness, lineType=cv2.LINE_AA)


def draw_frame(label: str, rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Draw one BGR frame of the given class."""
    H = W = size
    gx = np.linspace(-1, 1, W)[None, :]
    gy = np.linspace(-1, 1, H)[:, None]

    # 1 · Backlight panel: bright, uneven, vignetted
    base = rng.uniform(185, 235)
    bg = base + rng.uniform(-20, 20) * gx + rng.uniform(-20, 20) * gy - rng.uniform(10, 35) * (gx**2 + gy**2)
    tint = rng.uniform(0.93, 1.05, size=3)
    img = np.clip(np.stack([bg * t for t in tint], axis=-1), 0, 255)

    # 2 · Chamber geometry (the clip is never perfectly centred)
    cx = W / 2 + rng.uniform(-14, 14)
    cw = rng.uniform(0.36, 0.44) * W
    top = rng.uniform(0.08, 0.13) * H
    bottom = rng.uniform(0.86, 0.92) * H
    x0, x1 = cx - cw / 2, cx + cw / 2

    # plastic body: slightly darker and cooler than the panel
    body = img.copy()
    cv2.rectangle(body, (int(x0), int(top)), (int(x1), int(bottom)), (200, 205, 205), -1)
    img = cv2.addWeighted(img, 0.55, body, 0.45, 0)

    # 3 · Fluid (amber) filling the bottom of the chamber
    if label == "low_fluid":
        level = bottom - rng.uniform(0.02, 0.06) * H
    else:
        level = rng.uniform(0.52, 0.68) * H
    amber = (rng.uniform(35, 70), rng.uniform(125, 160), rng.uniform(180, 215))
    fluid = img.copy()
    cv2.rectangle(fluid, (int(x0) + 2, int(level)), (int(x1) - 2, int(bottom)), amber, -1)
    img = cv2.addWeighted(img, 0.25, fluid, 0.75, 0)
    # meniscus: a darker curved band on the fluid surface
    _ellipse(img, (cx, level), (cw / 2 - 3, rng.uniform(2, 4)), tuple(0.6 * a for a in amber), 2)

    # 4 · Chamber walls, spout on top, outlet tube at the bottom
    wall = tuple(rng.uniform(70, 110) for _ in range(3))
    t = int(rng.integers(3, 6))
    cv2.line(img, (int(x0), int(top)), (int(x0), int(bottom)), wall, t, cv2.LINE_AA)
    cv2.line(img, (int(x1), int(top)), (int(x1), int(bottom)), wall, t, cv2.LINE_AA)
    cv2.line(img, (int(x0), int(top)), (int(x1), int(top)), wall, t + 2, cv2.LINE_AA)
    cv2.line(img, (int(x0), int(bottom)), (int(x1), int(bottom)), wall, t + 1, cv2.LINE_AA)
    spout_w = rng.uniform(4, 7)
    spout_end = top + rng.uniform(0.07, 0.11) * H
    cv2.rectangle(img, (int(cx - spout_w), 0), (int(cx + spout_w), int(spout_end)), (60, 60, 60), -1)
    cv2.rectangle(img, (int(cx - 5), int(bottom)), (int(cx + 5), H), (90, 110, 130), -1)

    # a pendant drop hanging from the spout appears in every class — a distractor
    if rng.random() < 0.6:
        r = rng.uniform(3, 6)
        _ellipse(img, (cx, spout_end + r * 0.8), (r, r * 1.1), (45, 90, 120), -1)

    # 5 · The falling drop: dark refractive rim, amber core, specular highlight
    if label == "drop" or (label == "low_fluid" and rng.random() < 0.4):
        r = rng.uniform(5, 8)
        y_max = level - 2 * r - 4
        y_min = spout_end + 3 * r
        if y_max > y_min:
            dy = rng.uniform(y_min, y_max)
            dx = cx + rng.uniform(-3, 3)
            _ellipse(img, (dx, dy), (r, r * 1.25), (30, 55, 80), -1)
            _ellipse(img, (dx, dy), (r * 0.6, r * 0.8), amber, -1)
            _ellipse(img, (dx - r * 0.3, dy - r * 0.4), (r * 0.2, r * 0.25), (250, 250, 250), -1)

    img = np.clip(img, 0, 255).astype(np.uint8)

    # ---- camera damage ---------------------------------------------------------
    # tilt + zoom: the clip-on camera is mounted slightly crooked
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-6, 6), rng.uniform(0.95, 1.08))
    img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    # defocus
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.3))
    f = img.astype(np.float32)
    # dim ward at night: low exposure + low contrast
    if rng.random() < 0.45:
        alpha = rng.uniform(0.3, 0.6)
        f = f * alpha + rng.uniform(20, 70)
    # sensor noise
    f += rng.normal(0, rng.uniform(4, 14), f.shape)
    img = np.clip(f, 0, 255).astype(np.uint8)
    # dead / hot pixels (salt & pepper)
    if rng.random() < 0.5:
        p = rng.uniform(0.005, 0.03)
        m = rng.random(img.shape[:2])
        img[m < p / 2] = 0
        img[m > 1 - p / 2] = 255
    return img


def generate(out: Path, per_class: int = 120, seed: int = 42, size: int = 256, quality=(70, 95)) -> Path:
    """Write out/<class>/<class>_0000.jpg … Skips work if the same dataset already exists."""
    out = Path(out)
    stamp = out / ".stamp"
    key = f"v{GENERATOR_VERSION}-{per_class}-{seed}-{size}"
    if stamp.exists() and stamp.read_text() == key:
        return out
    rng = np.random.default_rng(seed)
    for label in CLASSES:
        d = out / label
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.jpg"):
            old.unlink()
        for i in range(per_class):
            img = draw_frame(label, rng, size)
            q = int(rng.integers(*quality))
            cv2.imwrite(str(d / f"{label}_{i:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, q])
    stamp.write_text(key)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    print(f"Wrote dataset to {generate(Path(a.out), a.per_class, a.seed, a.size)}")
