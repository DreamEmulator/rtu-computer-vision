"""Stage 1 · 📷 Digital Data — turn camera files (and your snap) into one consistent dataset.

Topic: Introduction to Image Processing (Tue 22.09) — pixels, channels, colour spaces, resize / crop / rotate.
Knobs: 01_image-processing/digital-data_prepare/action.yaml
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from stages.common import (ROOT, SNAP, ImageSet, StageReport, contact_sheet, fresh_stage_dir, load_config, run_stage,
                           save_images)

KEY = "digital_data"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
INTERPOLATION = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC}
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
COLOR = {"gray": cv2.COLOR_BGR2GRAY, "hsv": cv2.COLOR_BGR2HSV, "lab": cv2.COLOR_BGR2LAB, "ycrcb": cv2.COLOR_BGR2YCrCb}
MAX_SNAP_BYTES = 15 * 1024 * 1024


def prepare(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """The whole stage for one frame. Order matters: rotate → crop → resize → colour space."""
    img = img_bgr
    rot = int(cfg.get("rotate") or 0) % 360
    if rot:
        img = cv2.rotate(img, ROTATE[rot])
    if cfg.get("crop"):
        h, w = img.shape[:2]
        x0, y0, x1, y1 = cfg["crop"]
        img = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    width, height = cfg.get("size") or [128, 128]
    img = cv2.resize(img, (int(width), int(height)), interpolation=INTERPOLATION[cfg.get("interpolation", "area")])
    space = cfg.get("color_space", "gray")
    if space != "bgr":
        img = cv2.cvtColor(img, COLOR[space])
    if cfg.get("channel") is not None and img.ndim == 3:
        img = np.ascontiguousarray(img[..., int(cfg["channel"])])
    return img


def find_images(folder: Path) -> list[tuple[Path, str]]:
    """data/raw/<class>/<file> → [(path, class), ...]"""
    items = []
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        items += [(p, class_dir.name) for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMG_EXT]
    return items


def fetch_snap(src: str) -> tuple[np.ndarray, int]:
    """Step 0: the photo from sdux.tech/computer-vision (a URL) or a local file."""
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "cv-startup-pipeline"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read(MAX_SNAP_BYTES + 1)
    else:
        data = Path(src).expanduser().read_bytes()
    if len(data) > MAX_SNAP_BYTES:
        raise ValueError("The snap is larger than 15 MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Couldn't decode the snap. Send a jpg, png or webp.")
    return img, len(data)


def hardware() -> dict:
    """What this machine offers OpenCV: vector instructions (SIMD) and CUDA GPUs."""
    flags = [("SSE4.2", "CPU_SSE4_2"), ("AVX2", "CPU_AVX2"), ("AVX-512", "CPU_AVX_512F"), ("NEON", "CPU_NEON")]
    simd = [name for name, attr in flags if hasattr(cv2, attr) and cv2.checkHardwareSupport(getattr(cv2, attr))]
    try:
        gpus = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        gpus = 0
    return {"OpenCV": cv2.__version__, "CPU threads": cv2.getNumThreads(), "SIMD": simd or ["none"], "CUDA GPUs": gpus}


def shrink(img: np.ndarray, side: int = 384) -> np.ndarray:
    s = side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def main() -> None:
    cfg = load_config()
    c = cfg[KEY]
    report = StageReport(KEY, cfg)
    out_dir = fresh_stage_dir(KEY)

    if c.get("source", "synthetic") == "synthetic":
        from tools.synth_drips import generate
        folder = generate(ROOT / "data" / "synthetic", c.get("synthetic_per_class", 120), c.get("seed", 42))
        report.note("Using the synthetic day-zero dataset (source: synthetic)")
    else:
        folder = ROOT / (c.get("folder") or "data/raw")
    items = find_images(folder) if folder.exists() else []
    if not items:
        raise SystemExit(f"❌ No images found in {folder}. Expected {folder}/<class_name>/*.jpg")

    labels = [label for _, label in items]
    classes = sorted(set(labels))
    if min(labels.count(k) for k in classes) < 2:
        raise SystemExit("❌ Every class needs at least 2 images to split into train and test.")

    # Lock the test set on day one: stratified, seeded, never touched by training.
    train_idx, _ = train_test_split(np.arange(len(items)), test_size=c.get("test_size", 0.25),
                                    stratify=labels, random_state=c.get("seed", 42))
    train_set = set(train_idx.tolist())

    rows, images, unreadable, resolutions = [], [], 0, []
    t0 = time.perf_counter()
    for i, (path, label) in enumerate(items):
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            unreadable += 1
            report.note(f"Could not read {path.name}")
            continue
        resolutions.append(raw.shape[:2])
        rows.append({"file": f"{i:05d}_{label}.png", "label": label,
                     "split": "train" if i in train_set else "test", "source": os.path.relpath(path, ROOT)})
        images.append(prepare(raw, c))
    ms_per_frame = 1000 * (time.perf_counter() - t0) / max(1, len(images))

    space = "gray" if images[0].ndim == 2 else c.get("color_space", "gray")
    snap_raw = None
    snap_src = os.environ.get("SNAP_URL", "").strip()
    if snap_src:
        snap_raw, nbytes = fetch_snap(snap_src)
        t = time.perf_counter()
        snap_out = prepare(snap_raw, c)
        report.snap("camera resolution", f"{snap_raw.shape[1]}×{snap_raw.shape[0]}")
        report.snap("file size KB", nbytes / 1024)
        report.snap("prepare ms", 1000 * (time.perf_counter() - t))
        label = os.environ.get("SNAP_LABEL", "").strip()
        rows.append({"file": "snap.png", "label": label if label in classes else "unknown", "split": SNAP, "source": "snap"})
        images.append(snap_out)
        report.snap_image("camera", shrink(snap_raw), "bgr")
        report.snap_image("output", snap_out, space)

    save_images(KEY, ImageSet(rows, images, {"color_space": space, "classes": classes}))

    # Preview: raw camera frame vs what the pipeline works with
    raw_preview = [images[0]] * len(rows)
    for i, r in enumerate(rows):
        if r["split"] == SNAP:
            raw_preview[i] = cv2.resize(snap_raw, images[0].shape[1::-1])
        elif r["split"] == "test":
            raw = cv2.imread(str(ROOT / r["source"]), cv2.IMREAD_COLOR)
            raw_preview[i] = cv2.resize(raw, images[0].shape[1::-1]) if raw is not None else images[0]
    contact_sheet(out_dir / "preview.png", rows,
                  [("camera", raw_preview, "bgr"), (f"{space} {images[0].shape[1]}×{images[0].shape[0]}", images, space)],
                  "1 · Digital Data — raw camera frame vs pipeline input")

    data = [r for r in rows if r["split"] != SNAP]
    per_class = {k: sum(r["label"] == k for r in data) for k in classes}
    report.metric("classes", classes)
    report.metric("images", len(data))
    report.metric("images per class", [per_class[k] for k in classes])
    report.metric("train / test", [sum(r["split"] == "train" for r in data), sum(r["split"] == "test" for r in data)])
    report.metric("median raw resolution (h×w)", "×".join(str(int(v)) for v in np.median(resolutions, axis=0)))
    report.metric("output shape", "×".join(map(str, images[0].shape)))
    report.metric("colour space", space)
    report.metric("mean pixel value", float(np.mean([images[i].mean() for i in range(len(data))])))
    report.metric("unreadable files", unreadable)
    report.perf("ms per frame", ms_per_frame)
    for k, v in hardware().items():
        report.perf(k, v)

    report.gate("images per class", min(per_class.values()), min=c.get("gate_min_images_per_class"))
    report.gate("unreadable files", unreadable, max=c.get("gate_max_unreadable"))
    report.finish()


if __name__ == "__main__":
    run_stage(KEY, main)
