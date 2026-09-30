"""
✂️ Stage 4a · Thresholding Algorithms Sandbox
═════════════════════════════════════════════
One image in, one image out.

    📥 in   build/sandbox/stage_3_improving.png     the output of stage 3 (falls back to snaps/default_drip.jpg)
    📤 out  build/sandbox/stage_4a_threshold.png    a black-and-white mask → the input of 4b
    📝 log  build/sandbox/<unix time>_results_stage_4a.md

Run it from the repository root (or press ▶ in PyCharm):

    python 03_image-segmentation/sandboxes/sandbox_4a_threshold.py
    python 03_image-segmentation/sandboxes/sandbox_4a_threshold.py path/to/your/photo.jpg

👾 The Robot sorts every pixel into one of two bins: object (white) or background (black).
   The thresholding algorithm is its brain: the rule that decides which bin a pixel lands in.
   Global, Otsu, Triangle and Adaptive are four different brains.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#    🎛️  TINKER ZONE — change one value, run again, compare the two results files
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
METHOD = "triangle"   # "global"   one fixed T for every pixel (you pick it below)
#                       "otsu"     tries all 256 T's, keeps the one that best splits TWO hills (Otsu, 1979)
#                       "triangle" made for ONE big hill with a tail, like our grey wall (Zack et al., 1977)
#                       "adaptive" a different T for every neighbourhood: fair under uneven light
GLOBAL_T = 127        # only for "global": 0 = black … 255 = white
ADAPTIVE_BLOCK = 31   # only for "adaptive": the neighbourhood size in pixels (odd number)
ADAPTIVE_C = 5        # only for "adaptive": how much darker than its neighbours a pixel must be
INVERT = True         # True: DARK pixels are the object (the drip chamber is darker than the wall)
MAX_SIDE = 800        # shrink bigger photos first, so kernel sizes in 4b mean the same on every photo
SPECK_AREA = 20       # blobs smaller than this many pixels count as specks (salt)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    # 1. 📥 Load the output of the previous step (stage 3 · Improving)
    src = pick_input(OUT / "stage_3_improving.png", fallback=ROOT / "snaps" / "default_drip.jpg")
    img = read(src)
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:                                   # e.g. a 2252×4000 phone photo → 450×800
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save(gray, "stage_4_photo.png")                            # 4c and 4d draw their overlays on this

    # 2. 🧠 Threshold: the Robot's brain decides which bin every pixel lands in
    mode = cv2.THRESH_BINARY_INV if INVERT else cv2.THRESH_BINARY
    t0 = time.perf_counter()
    if METHOD == "global":
        T, mask = cv2.threshold(gray, GLOBAL_T, 255, mode)                    # one rule for everyone
    elif METHOD == "otsu":
        T, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)         # the 0 is ignored: Otsu picks T
    elif METHOD == "triangle":
        T, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_TRIANGLE)     # the 0 is ignored: Triangle picks T
    elif METHOD == "adaptive":
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode,
                                     ADAPTIVE_BLOCK, ADAPTIVE_C)              # compare with the local mean − C
        T = None                                                              # a different T per neighbourhood
    else:
        raise SystemExit(f'👾 I don\'t know METHOD = "{METHOD}". Pick global, otsu, triangle or adaptive.')
    ms = 1000 * (time.perf_counter() - t0)
    out = save(mask, "stage_4a_threshold.png")

    # 3. 📊 Measure what happened
    fg = 100 * np.count_nonzero(mask) / mask.size                            # share of pixels in the object bin
    otsu_T, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    eta = separability(gray, otsu_T if T is None else T)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]                                       # label 0 is the background
    blobs, specks = len(areas), int((areas < SPECK_AREA).sum())
    biggest = 100 * areas.max() / max(1, areas.sum()) if blobs else 0.0

    # 4. 🚦 Judge the numbers
    if fg < 0.5:
        fg_level, fg_why = "bad", "Almost nobody got in: the rule is too strict. Move T, or flip INVERT."
    elif fg > 95:
        fg_level, fg_why = "bad", "Everybody got in: the mask is all object. Flip INVERT, or move T."
    elif fg > 60:
        fg_level, fg_why = "check", "More object than background. For a drip chamber in a frame that's suspicious: is INVERT right?"
    else:
        fg_level, fg_why = "good", "Some pixels in, most out. The pipeline's gates fail below 0.5 % (empty) and above 95 % (full)."
    eta_level = "good" if eta >= 0.8 else "check" if eta >= 0.5 else "bad"
    speck_share = 100 * specks / max(1, blobs)

    metrics = [
        ("🎯", "Threshold T", "per neighbourhood" if T is None else f"{T:.0f}", "info",
         "The brightness where the Robot switches bins. "
         + ("Adaptive uses a different T for every pixel's neighbourhood." if T is None
            else f"For comparison, Otsu would pick {otsu_T:.0f}.")),
        ("⚪", "Foreground", f"{fg:.1f} %", fg_level, fg_why),
        ("⛰️", "Separability η", f"{eta:.2f}" + (" (at Otsu's T)" if T is None else ""),
         "info" if T is None else eta_level,
         "0 → 1, higher is better: how cleanly one T splits the histogram into two groups (Otsu's own score). "
         "Below 0.5 the histogram has no clean valley, so one global T will struggle: try triangle or adaptive."),
        ("🫧", "Blobs", f"{blobs}", "info",
         "Separate white islands. Fewer, bigger blobs means cleaner objects; hundreds means salt."),
        ("🧂", "Specks", f"{specks} ({speck_share:.0f} % of blobs)", "check" if speck_share > 50 else "good",
         f"Blobs under {SPECK_AREA} px. Lower is better. Specks aren't fatal: 4b's opening sands them off."),
        ("🐘", "Biggest blob", f"{biggest:.0f} % of foreground", "info",
         "High = one dominant object (hopefully the chamber). Low = the foreground is scattered."),
        ("⏱️", "Time", f"{ms:.2f} ms ({1000 / max(ms, 1e-3):.0f} fps)", "info",
         "Lower is better: a clip-on camera has to keep up with the drops. Thresholding is about the cheapest step in the pipeline."),
    ]

    tips = []
    if fg_level == "bad":
        tips.append("Flip `INVERT` first: it's the most common reason for an empty or full mask.")
    if METHOD in ("global", "otsu") and eta < 0.5:
        tips.append('One big hill and no valley: try `METHOD = "triangle"` or `"adaptive"`.')
    if METHOD == "global":
        tips.append(f"Otsu would have chosen T = {otsu_T:.0f}. Try `GLOBAL_T = {otsu_T:.0f}` and compare.")
    if METHOD == "adaptive":
        tips.append("Make `ADAPTIVE_BLOCK` bigger (51, 101): you get fewer rims, and it behaves more like one global T.")
    tips += ["Try all four methods on the same photo, then compare the results files side by side.",
             "Next: `sandbox_4b_morphology.py` repairs this mask."]

    story = (f"👾 I took **{src.name}**, turned it grey ({gray.shape[1]}×{gray.shape[0]} px) and sorted every pixel "
             f"into two bins with the **{METHOD}** brain. "
             + ("Dark pixels went into the object bin (white), because `INVERT` is on. " if INVERT
                else "Bright pixels went into the object bin (white). ")
             + f"**{fg:.1f} %** of the pixels ended up as object.")
    log = write_results("4a", "✂️ Stage 4a · Thresholding", story,
                        files=[("📥 input", rel(src)), ("📤 output", rel(out))],
                        knobs={"METHOD": METHOD, "GLOBAL_T": GLOBAL_T, "ADAPTIVE_BLOCK": ADAPTIVE_BLOCK,
                               "ADAPTIVE_C": ADAPTIVE_C, "INVERT": INVERT, "MAX_SIDE": MAX_SIDE},
                        metrics=metrics, tips=tips)
    report(metrics, [out], log)


def separability(gray: np.ndarray, T: float) -> float:
    """Otsu's η = between-group variance / total variance for the split at T. 1.0 = two perfectly separate hills."""
    g = gray.astype(np.float64).ravel()
    lo, hi = g[g <= T], g[g > T]
    if lo.size == 0 or hi.size == 0 or g.var() == 0:
        return 0.0
    w0, w1 = lo.size / g.size, hi.size / g.size
    return float(w0 * w1 * (lo.mean() - hi.mean()) ** 2 / g.var())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔒 PLUMBING — the same in every sandbox, no need to touch ━━━━━━━━━━━━━━━━━━━━━━━━━━━

def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for folder in (here, *here.parents):
        if (folder / "run_pipeline.py").exists():
            return folder
    return Path.cwd()


ROOT = repo_root()
OUT = ROOT / "build" / "sandbox"   # build/ is in .gitignore: your experiments stay on your machine


def pick_input(default: Path, fallback: Path | None = None, hint: str = "") -> Path:
    """The path on the command line, else the previous step's output, else the fallback."""
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return Path(sys.argv[1]).expanduser()
    if default.exists():
        return default
    if fallback is not None and fallback.exists():
        print(f"👾 {rel(default)} isn't there yet, so I'm using {rel(fallback)}.")
        return fallback
    raise SystemExit(f"👾 I need {rel(default)} first. {hint}")


def read(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    if not Path(path).is_file():
        raise SystemExit(f"👾 There's no file at {path}. Check the path, or run the previous sandbox first.")
    img = cv2.imread(str(path), flags)
    if img is None:
        raise SystemExit(f"👾 I can't read {path}. Is it an image, and is the path right?")
    return img


def read_mask(path: Path) -> np.ndarray:
    """Masks are black (0) and white (255). Re-binarise, in case an editor or a JPEG blurred them."""
    return np.where(read(path, cv2.IMREAD_GRAYSCALE) > 127, 255, 0).astype(np.uint8)


def read_photo(shape: tuple) -> np.ndarray | None:
    """The grey photo 4a worked on, for overlays. None if it's missing or a different size."""
    path = OUT / "stage_4_photo.png"
    photo = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None
    return photo if photo is not None and photo.shape == shape[:2] else None


def save(img: np.ndarray, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    cv2.imwrite(str(path), img)
    return path


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


LIGHTS = {"good": "🟢", "check": "🟠", "bad": "🔴", "info": "ℹ️"}


def write_results(stage: str, title: str, story: str, files: list, knobs: dict, metrics: list, tips: list,
                  extra: str = "") -> Path:
    now = time.time()
    lines = [f"# {title} · results", "",
             f"🕒 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))} · unix `{int(now)}`", "",
             "## 👾 What the Robot did", "", story, "",
             "## 📂 Files", "", "| | |", "|---|---|", *[f"| {k} | `{v}` |" for k, v in files], "",
             "## 🎛️ Knobs used", "", "| Knob | Value |", "|---|---|", *[f"| `{k}` | `{v}` |" for k, v in knobs.items()], "",
             "## 📊 Metrics", "", "| | Metric | Value | | What it tells you |", "|---|---|---|---|---|",
             *[f"| {e} | {m} | **{v}** | {LIGHTS[lvl]} | {why} |" for e, m, v, lvl, why in metrics], "",
             "🟢 looks right · 🟠 worth a look · 🔴 something is off · ℹ️ for your information", "",
             *([extra, ""] if extra else []),
             "## 💡 Try next", "", *[f"- {t}" for t in tips], ""]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{int(now)}_results_stage_{stage}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def report(metrics: list, outputs: list, log: Path) -> None:
    print()
    for e, m, v, lvl, _ in metrics:
        print(f"  {LIGHTS[lvl]} {e} {m}: {v}")
    for path in outputs:
        print(f"  📤 {rel(path)}")
    print(f"  📝 {rel(log)}\n")


if __name__ == "__main__":
    main()
