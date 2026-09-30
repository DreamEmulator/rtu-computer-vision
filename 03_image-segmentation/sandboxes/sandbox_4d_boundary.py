"""
✂️ Stage 4d · Boundary Extraction Sandbox
═════════════════════════════════════════
One image in, one image out.

    📥 in   build/sandbox/stage_4c_contours.png    the filled objects from 4c
    📤 out  build/sandbox/stage_4d_boundary.png    only their crust, white on black → the input of stage 5
    👀 look build/sandbox/stage_4d_overlay.png     the crust painted on the photo
    📝 log  build/sandbox/<unix time>_results_stage_4d.md

Run it from the repository root (or press ▶ in PyCharm):

    python 03_image-segmentation/sandboxes/sandbox_4d_boundary.py
    python 03_image-segmentation/sandboxes/sandbox_4d_boundary.py path/to/any_mask.png

👾 The cookie crust. Take a solid baked cookie and scoop out the soft centre: what's left is the crust.
   The Robot does exactly that with 4b's tools. It makes a slightly eroded (shrunk) copy of the object
   and subtracts it from the original: β(A) = A − (A ⊖ B). Only the crust is left.

   Contours (4c) give you the outline as a list of (x, y) points, which is good for measuring.
   Boundaries give you the outline as pixels in an image, which is good for drawing and overlays.
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
METHOD = "inner"      # "inner"    the crust inside the cookie:  A − erode(A)
#                       "outer"    a ring of icing just outside:  dilate(A) − A
#                       "gradient" both at once, a thicker line:   dilate(A) − erode(A)
KERNEL_SHAPE = "rect" # "rect", "ellipse" or "cross": the shape of the scoop
KERNEL_SIZE = 3       # odd number: for inner and outer, 3 gives a crust about 1 px thick, 5 about 2 px, 7 about 3 px
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOLS = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}


def main() -> None:
    # 1. 📥 Load the filled objects from 4c
    src = pick_input(OUT / "stage_4c_contours.png", hint="Run sandbox_4c_contours.py first.")
    mask = read_mask(src)
    if not mask.any():
        raise SystemExit("👾 The input has no white pixels, so there's no cookie to take the crust from. Check 4a–4c.")
    if KERNEL_SHAPE not in TOOLS:
        raise SystemExit(f'👾 I don\'t have a "{KERNEL_SHAPE}" scoop. Pick rect, ellipse or cross.')
    kernel = cv2.getStructuringElement(TOOLS[KERNEL_SHAPE], (KERNEL_SIZE, KERNEL_SIZE))

    # 2. 🍪 Scoop out the centre, keep the crust
    t0 = time.perf_counter()
    if METHOD == "inner":
        crust = cv2.subtract(mask, cv2.erode(mask, kernel))                  # the cookie minus a shrunk cookie
    elif METHOD == "outer":
        crust = cv2.subtract(cv2.dilate(mask, kernel), mask)                 # a grown cookie minus the cookie
    elif METHOD == "gradient":
        crust = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)          # grown minus shrunk
    else:
        raise SystemExit(f'👾 I don\'t know METHOD = "{METHOD}". Pick inner, outer or gradient.')
    ms = 1000 * (time.perf_counter() - t0)
    out = save(crust, "stage_4d_boundary.png")

    # 3. 👀 Paint the crust on the photo from 4a (or on the mask, if the photo doesn't match)
    photo = read_photo(mask.shape)
    base = photo if photo is not None else (mask // 3 + 40).astype(np.uint8)
    vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    vis[crust > 0] = (31, 140, 240)                                          # orange crust (BGR)
    look = save(vis, "stage_4d_overlay.png")

    # 4. 📊 Measure the crust
    object_px, crust_px = np.count_nonzero(mask), np.count_nonzero(crust)
    outlines, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)   # outer outlines + holes
    perimeter = sum(cv2.arcLength(c, True) for c in outlines)
    thickness = crust_px / perimeter if perimeter else 0.0
    share = 100 * crust_px / max(1, object_px)
    pieces = cv2.connectedComponents(crust, connectivity=8)[0] - 1
    expected = len(outlines)

    # 5. 🚦 Judge the numbers
    if pieces == expected:
        piece_level, piece_why = "good", "One closed crust per outline: every object and hole got a complete border."
    elif pieces > expected:
        piece_level, piece_why = "check", ("More pieces than outlines: some crusts broke. Either an object is thinner "
                                           "than the scoop (try a smaller KERNEL_SIZE), or it touches the edge of the "
                                           "image, where OpenCV leaves the crust open.")
    else:
        piece_level, piece_why = "check", ("Fewer pieces than outlines: crusts touch and merge, where objects or "
                                           "holes lie closer together than the crust is thick.")
    metrics = [
        ("🍪", "Object pixels", f"{object_px}", "info", "The whole cookie: every white pixel of the input."),
        ("🥧", "Crust pixels", f"{crust_px}", "info", "What's left after scooping. This is the output image."),
        ("📏", "Crust thickness", f"{thickness:.1f} px", "info",
         "Crust pixels ÷ outline length. About 1 px = a crisp line for precise drawing. "
         "Thicker is easier to see and survives small shifts, but blurs detail."),
        ("🫓", "Crust share", f"{share:.1f} % of the object", "check" if share > 60 else "good",
         "Low = solid objects with a thin rim (what we want from 4c). Near 100 % = the objects are "
         "already thin lines: was FILL_HOLES off in 4c?"),
        ("🧩", "Crust pieces", f"{pieces} (outlines: {expected})", piece_level, piece_why),
        ("⏱️", "Time", f"{ms:.2f} ms ({1000 / max(ms, 1e-3):.0f} fps)", "info",
         "Lower is better. One erosion or dilation plus a subtraction: cheap."),
    ]

    tips = ['Try `METHOD = "outer"`: the crust now sits outside the object. When would you want that?',
            "Set `KERNEL_SIZE` to 3, 5 and 7 and watch the thickness go up by about 1 px per step.",
            "Compare stage_4d_overlay.png with stage_4c_overlay.png: same outline, once as pixels and once as points.",
            "Next stop: stage 5 · Extracting turns these objects into numbers a classifier can learn from."]

    story = (f"👾 I took the objects in **{src.name}** and scooped out their centres with a "
             f"**{KERNEL_SIZE}×{KERNEL_SIZE} {KERNEL_SHAPE}** scoop ({METHOD} method). "
             f"**{crust_px}** crust pixels are left of **{object_px}**: a crust about **{thickness:.1f} px** thick "
             f"in {pieces} piece{'s' if pieces != 1 else ''}.")
    log = write_results("4d", "✂️ Stage 4d · Boundary Extraction", story,
                        files=[("📥 input", rel(src)), ("📤 output", rel(out)), ("👀 overlay", rel(look))],
                        knobs={"METHOD": METHOD, "KERNEL_SHAPE": KERNEL_SHAPE, "KERNEL_SIZE": KERNEL_SIZE},
                        metrics=metrics, tips=tips)
    report(metrics, [out, look], log)


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
