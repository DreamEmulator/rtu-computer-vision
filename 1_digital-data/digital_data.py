"""
📷 Stage 1 · Digital Data — can we turn whatever the camera gives us into one consistent format?
═══════════════════════════════════════════════════════════════════════════════════════════════
Lecture: Introduction to Image Processing · Tue 22.09 · Why this stage exists: README.md · The values: action.yaml

This is the file CI runs for this stage. Press ▶ in PyCharm (or `python 1_digital-data/digital_data.py`):

    build/digital_data/report.md          the values you passed in, every number explained, the gates
    build/digital_data/before_after.png   your image for analyzing and one test frame per class: camera → pipeline input

Change one value in action.yaml, press ▶ again, and the report says what changed since your last run.

👾 Pixels are numbers. A camera frame is a NumPy array of height × width × 3 channels (blue, green, red in
   OpenCV), every value 0 … 255. Every later stage needs the SAME kind of array, whatever camera or file
   format a frame came from. So this stage, for every frame and for your image for analyzing:
     1. rotate      a sideways-mounted camera (a transpose and a flip: no maths at all)
     2. crop        keep the part of the frame that matters (a NumPy view: no pixel is copied)
     3. resize      every frame the same size. Shrinking throws pixels away: Nyquist & Shannon say a detail
                    must be at least two pixels wide to survive. A 6-pixel drop shrunk 8× is gone.
     4. colour      grey (one channel), or a colour space where what matters stands out: in HSV or Lab one
                    channel can separate amber fluid from air better than any of B, G, R.
   And it locks the test set, on day one: a share of the frames the model will never learn from.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # 🔒 finds stages/

import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from stages.common import ROOT, SNAP
from stages.stage import Stage, picture, run   # 🔒 the plumbing: values, saving, report, gates

KEY = "digital_data"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
INTERPOLATION = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC}
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
COLOR = {"gray": cv2.COLOR_BGR2GRAY, "hsv": cv2.COLOR_BGR2HSV, "lab": cv2.COLOR_BGR2LAB, "ycrcb": cv2.COLOR_BGR2YCrCb}
MAX_SNAP_BYTES = 15 * 1024 * 1024


def prepare(img_bgr: np.ndarray, stage_values: dict, steps: list | None = None) -> np.ndarray:
    """The whole stage for one frame. Order matters: rotate → crop → resize → colour space."""
    img = img_bgr
    picture(steps, "camera", shrink(img), "bgr")
    rot = int(stage_values.get("rotate") or 0) % 360
    if rot:
        img = cv2.rotate(img, ROTATE[rot])
        picture(steps, f"rotate {rot}°", shrink(img), "bgr")
    if stage_values.get("crop"):
        h, w = img.shape[:2]
        x0, y0, x1, y1 = stage_values["crop"]
        img = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        picture(steps, "crop", shrink(img), "bgr")
    width, height = stage_values.get("size") or [128, 128]
    img = cv2.resize(img, (int(width), int(height)), interpolation=INTERPOLATION[stage_values.get("interpolation", "area")])
    picture(steps, f"resize {width}×{height}", img, "bgr")
    space = stage_values.get("color_space", "gray")
    if space != "bgr":
        img = cv2.cvtColor(img, COLOR[space])
    if stage_values.get("channel") is not None and img.ndim == 3:
        img = np.ascontiguousarray(img[..., int(stage_values["channel"])])
    picture(steps, f"out: {space}" + (f" channel {stage_values['channel']}" if stage_values.get("channel") is not None else ""),
            img, "gray" if img.ndim == 2 else space)
    return img


def find_images(folder: Path) -> list[tuple[Path, str]]:
    """images_for_training/<class>/<file> → [(path, class), ...]"""
    items = []
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        items += [(p, class_dir.name) for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMG_EXT]
    return items


def groups_of(items: list[tuple[Path, str]], how: str) -> list[str]:
    """Which frames must stay together on one side of the split.

    file    every file on its own (fine for separate photos)
    prefix  everything before the last "_" in the file name: cow-pasture_0012.jpg → cow-pasture,
            so all frames of one video (tools/frames_from_video.py) stay together
    """
    if how == "file":
        return [str(p) for p, _ in items]
    if how == "prefix":
        return [f"{label}/{p.stem.rsplit('_', 1)[0]}" for p, label in items]
    raise SystemExit(f"❌ Unknown group_by '{how}'. Choose file | prefix")


def split(labels: list[str], groups: list[str], test_size: float, seed: int) -> np.ndarray:
    """Indices of the training frames. Stratified (every class in train and test), seeded, and groups never cross."""
    if len(set(groups)) == len(groups):                   # every frame its own group: the classic split
        train_idx, _ = train_test_split(np.arange(len(labels)), test_size=test_size, stratify=labels, random_state=seed)
        return train_idx
    per_class = {k: len({g for g, l in zip(groups, labels) if l == k}) for k in sorted(set(labels))}
    few = [k for k, n in per_class.items() if n < 2]
    if few:
        raise SystemExit(f"❌ {', '.join(few)} {'comes' if len(few) == 1 else 'come'} from one video only, so "
                         f"{'it' if len(few) == 1 else 'they'} can't be in both train and test. Record a second video per class, or set group_by: file (and accept the leak).")
    folds = StratifiedGroupKFold(n_splits=max(2, min(round(1 / test_size), min(per_class.values()))),
                                 shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(np.zeros(len(labels)), labels, groups):
        if {labels[i] for i in test_idx} == set(labels):  # the first fold with every class in it
            return train_idx
    raise SystemExit("❌ No split puts every class in the test set. Record more videos per class.")


def fetch_snap(src: str) -> tuple[np.ndarray, int]:
    """Step 0: the photo from sdux.tech/computer-vision (a URL) or a local file."""
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "cv-startup-pipeline"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read(MAX_SNAP_BYTES + 1)
    else:
        data = Path(src).expanduser().read_bytes()
    if len(data) > MAX_SNAP_BYTES:
        raise ValueError("The image for analyzing is larger than 15 MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Couldn't decode the image for analyzing. Send a jpg, png or webp.")
    return img, len(data)

def default_snap() -> str:
    """No snap_url given? Use the image in images_for_analyzing/ (the last one by name if there are several)."""
    folder = ROOT / "images_for_analyzing"
    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT) if folder.is_dir() else []
    return str(photos[-1]) if photos else ""

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
    stage = Stage(KEY)                           # the values from action.yaml
    stage_values, report = stage.values, stage.report

    if stage_values.get("source", "synthetic") == "synthetic":
        from tools.synth_drips import generate
        folder = generate(ROOT / "build" / "synthetic", stage_values.get("synthetic_per_class", 120), stage_values.get("seed", 42))
        report.note("Using the synthetic day-zero dataset (source: synthetic)")
    else:
        folder = ROOT / (stage_values.get("folder") or "images_for_training")
    items = find_images(folder) if folder.exists() else []
    if not items:
        raise SystemExit(f"❌ No images found in {folder}. Expected {folder}/<class_name>/*.jpg")

    labels = [label for _, label in items]
    classes = sorted(set(labels))
    if min(labels.count(k) for k in classes) < 2:
        raise SystemExit("❌ Every class needs at least 2 images to split into train and test.")

    # Lock the test set on day one: stratified, seeded, never touched by training, groups never split.
    groups = groups_of(items, stage_values.get("group_by") or "file")
    train_idx = split(labels, groups, stage_values.get("test_size", 0.25), stage_values.get("seed", 42))
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
        images.append(prepare(raw, stage_values))
    ms_per_frame = 1000 * (time.perf_counter() - t0) / max(1, len(images))

    space = "gray" if images[0].ndim == 2 else stage_values.get("color_space", "gray")
    snap_raw = None
    snap_src = os.environ.get("SNAP_URL", "").strip() or default_snap()
    if snap_src:
        snap_raw, nbytes = fetch_snap(snap_src)
        t = time.perf_counter()
        snap_out = prepare(snap_raw, stage_values)
        report.snap("camera resolution", f"{snap_raw.shape[1]}×{snap_raw.shape[0]}")
        report.snap("file size KB", nbytes / 1024)
        report.snap("prepare ms", 1000 * (time.perf_counter() - t))
        label = os.environ.get("SNAP_LABEL", "").strip()
        rows.append({"file": "snap.png", "label": label if label in classes else "unknown", "split": SNAP, "source": "snap"})
        images.append(snap_out)
        report.snap_image("camera", shrink(snap_raw), "bgr")
        report.snap_image("output", snap_out, space)

    stage.save(images, {"color_space": space, "classes": classes}, rows=rows)

    # 👀 before_after.png: the camera frame and every step, for your image for analyzing and one test frame per class
    stage.picture_rows(rows)
    for i, r in enumerate(rows):
        if (steps := stage.steps(i)) is not None:
            raw = snap_raw if r["split"] == SNAP else cv2.imread(str(ROOT / r["source"]), cv2.IMREAD_COLOR)
            prepare(raw, stage_values, steps)

    data = [r for r in rows if r["split"] != SNAP]
    per_class = {k: sum(r["label"] == k for r in data) for k in classes}
    report.metric("classes", classes)
    report.metric("images", len(data))
    report.metric("images per class", [per_class[k] for k in classes],
                  "Frames per class, in the order of classes. Unbalanced classes teach a model to guess the big one.",
                  "good" if max(per_class.values()) <= 2 * min(per_class.values()) else "check")
    report.metric("train / test", [sum(r["split"] == "train" for r in data), sum(r["split"] == "test" for r in data)],
                  "The test frames are locked away today; every later stage keeps its hands off them.")
    if (stage_values.get("group_by") or "file") != "file":
        report.metric(f"groups per class ({stage_values['group_by']})",
                      [len({g for g, l in zip(groups, labels) if l == k}) for k in classes])
    report.metric("median raw resolution (h×w)", "×".join(str(int(v)) for v in np.median(resolutions, axis=0)))
    report.metric("output shape", "×".join(map(str, images[0].shape)),
                  "What every later stage receives: height × width (× channels). Smaller is faster, until the drop is gone.")
    report.metric("colour space", space)
    report.metric("mean pixel value", float(np.mean([images[i].mean() for i in range(len(data))])))
    report.metric("unreadable files", unreadable, "Files OpenCV couldn't decode.", "good" if unreadable == 0 else "bad")
    report.perf("ms per frame", ms_per_frame)
    for k, v in hardware().items():
        report.perf(k, v)

    stage.tip("`size: \"[32, 32]\"`: look at the drop in before_after.png. Is it still there? Put it back afterwards.")
    stage.tip("`color_space: \"hsv\"` with `channel: \"2\"`, then `\"lab\"` with `channel: \"0\"`. Which looks most like grey, and why?")
    if stage_values.get("source", "synthetic") == "synthetic":
        stage.tip("Your own classes: images in images_for_training/<class>/ and `source: \"folder\"`. See how-to/add-my-own-classes.md.")

    report.gate("images per class", min(per_class.values()), min=stage_values.get("gate_min_images_per_class"))
    report.gate("unreadable files", unreadable, max=stage_values.get("gate_max_unreadable"))
    stage.finish("1 · Digital Data — camera frame → pipeline input")


if __name__ == "__main__":
    run(KEY, main)
