"""
✂️ Stage 4c · Contour Detection Sandbox
═══════════════════════════════════════
One image in, one image out.

    📥 in   build/sandbox/stage_4b_morphology.png   the repaired mask from 4b
    📤 out  build/sandbox/stage_4c_contours.png     only the objects we keep, filled → the input of 4d
    👀 look build/sandbox/stage_4c_overlay.png      every kept contour drawn on the photo
    📝 log  build/sandbox/<unix time>_results_stage_4c.md

Run it from the repository root (or press ▶ in PyCharm):

    python 03_image-segmentation/sandboxes/sandbox_4c_contours.py
    python 03_image-segmentation/sandboxes/sandbox_4c_contours.py path/to/any_mask.png

👾 Tracing your hand: put your hand on paper and follow its outline without lifting the pencil.
   The Robot does the same for every white blob (Suzuki & Abe, 1985: follow the border pixel by pixel
   until you're back where you started). Instead of "a block of white pixels" it now has a list of
   (x, y) points around each object, so it can count objects and measure them.
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
MODE = "external"   # "external" only the outer outline of each object (what the pipeline uses)
#                     "list"     every outline, holes too, no family tree
#                     "ccomp"    two levels: outer outlines and their holes
#                     "tree"     every outline with the full family tree (the Kinder Surprise)
POINTS = "simple"   # "none" store every border pixel · "simple" store only the corners of straight runs
MIN_AREA = 150      # outlines enclosing fewer pixels than this are specks: throw them away
FILL_HOLES = True   # True: colour in everything inside a kept outline (Jordan, 1887: a closed curve has an inside)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODES = {"external": cv2.RETR_EXTERNAL, "list": cv2.RETR_LIST, "ccomp": cv2.RETR_CCOMP, "tree": cv2.RETR_TREE}
APPROX = {"none": cv2.CHAIN_APPROX_NONE, "simple": cv2.CHAIN_APPROX_SIMPLE}


def main() -> None:
    # 1. 📥 Load the repaired mask from 4b
    src = pick_input(OUT / "stage_4b_morphology.png", hint="Run sandbox_4b_morphology.py first.")
    mask = read_mask(src)
    if MODE not in MODES or POINTS not in APPROX:
        raise SystemExit("👾 MODE must be external, list, ccomp or tree, and POINTS must be none or simple.")

    # 2. ✏️ Trace every outline
    t0 = time.perf_counter()
    contours, hierarchy = cv2.findContours(mask, MODES[MODE], APPROX[POINTS])
    # 3. 🧹 Throw away the specks
    kept = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    # 4. 🎨 Paint the objects we keep: filled, or only where the mask already was white
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, kept, -1, 255, thickness=cv2.FILLED)
    objects = filled if FILL_HOLES else cv2.bitwise_and(mask, filled)
    ms = 1000 * (time.perf_counter() - t0)
    out = save(objects, "stage_4c_contours.png")

    # 5. 📏 Measure every kept object, biggest first
    kept.sort(key=cv2.contourArea, reverse=True)
    rows = [measure(c, mask.size) for c in kept]
    points = sum(len(c) for c in contours)
    children = int((hierarchy[0][:, 3] != -1).sum()) if hierarchy is not None else 0
    dropped = len(contours) - len(kept)

    # 6. 👀 Draw on the photo from 4a (or on the mask, if the photo doesn't match)
    photo = read_photo(mask.shape)
    base = photo if photo is not None else (mask // 3 + 40).astype(np.uint8)
    vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, kept, -1, (31, 140, 240), 2)                         # orange: every kept object
    if kept:
        x, y, w, h = cv2.boundingRect(kept[0])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (200, 110, 30), 2)          # blue: the biggest one's box
        cv2.drawContours(vis, [cv2.convexHull(kept[0])], -1, (90, 200, 90), 1)  # green: its rubber band (hull)
        cx, cy = rows[0]["centroid"]
        cv2.circle(vis, (round(cx), round(cy)), 5, (90, 200, 90), -1)            # green dot: its centre
    look = save(vis, "stage_4c_overlay.png")

    # 7. 🚦 Judge the numbers
    n = len(kept)
    if n == 0:
        n_level, n_why = "bad", "Nothing left to measure. Lower MIN_AREA, or go back to 4a/4b."
    elif n <= 10:
        n_level, n_why = "good", "A handful of clear objects: easy to pick the drip chamber from."
    else:
        n_level, n_why = "check", "More than 10 objects: for a drip frame that's probably noise. Raise MIN_AREA, or clean harder in 4b."
    big = rows[0] if rows else None
    metrics = [
        ("✏️", "Outlines traced", f"{len(contours)}", "info",
         f"Every outline the Robot found with MODE = {MODE} (holes count as outlines in list, ccomp and tree)."),
        ("🧹", "Thrown away", f"{dropped} ({100 * dropped / max(1, len(contours)):.0f} %)", "info",
         f"Outlines enclosing less than {MIN_AREA} px². A high share means 4b left a lot of salt."),
        ("📦", "Objects kept", f"{n}", n_level, n_why),
        ("🪆", "Nested outlines", f"{children}", "info",
         "Outlines inside another outline (holes, or objects in holes). Always 0 with MODE = external."),
        ("📍", "Points stored", f"{points}", "info",
         'Lower is cheaper. POINTS = "simple" keeps only corners of straight runs; "none" keeps every border pixel.'),
    ]
    if big:
        frame_level = ("bad" if big["frame"] > 90 else "check" if big["frame"] > 50 or big["frame"] < 0.5
                       else "good")
        metrics += [
            ("🐘", "Biggest object", f"{big['area']:.0f} px² ({big['frame']:.1f} % of the frame)", frame_level,
             "Over half the frame is suspicious and over 90 % is almost certainly the background: "
             "flip INVERT in 4a. Under 0.5 % means even the biggest object is tiny."),
            ("⭕", "Its circularity", f"{big['circ']:.2f}", "info",
             "4πA / P², 1.0 = a perfect circle. A square scores 0.79, a 1:4 rectangle 0.50, ragged outlines far less. "
             "A drip chamber is a tall rectangle, so expect about 0.5 or less."),
            ("🪢", "Its solidity", f"{big['solid']:.2f}", "good" if big["solid"] >= 0.9 else "check",
             "Area / area of its convex hull (a rubber band around it), higher = smoother. "
             "Below 0.9 the outline has dents: concave on purpose, or bitten by noise?"),
            ("📐", "Its aspect ratio", f"{big['aspect']:.2f}", "info",
             "Width / height of its box: under 1 is tall, over 1 is wide. The drip chamber is tall; the pole is very wide."),
        ]
    metrics.append(("⏱️", "Time", f"{ms:.2f} ms ({1000 / max(ms, 1e-3):.0f} fps)", "info",
                    "Lower is better. Grows with the number and length of outlines."))

    table = ["## 🏷️ Objects, biggest first", "",
             "| # | Area px² | % of frame | Circularity | Solidity | Aspect w/h | Box x, y, w, h |",
             "|---|---|---|---|---|---|---|"]
    table += [f"| {i + 1} | {r['area']:.0f} | {r['frame']:.2f} | {r['circ']:.2f} | {r['solid']:.2f} | "
              f"{r['aspect']:.2f} | {', '.join(map(str, r['box']))} |" for i, r in enumerate(rows[:10])]
    if len(rows) > 10:
        table.append(f"\n…and {len(rows) - 10} smaller ones.")

    tips = []
    if n > 10:
        tips.append("Raise `MIN_AREA` until only real objects are left. How many pixels is the drip chamber?")
    if big and big["frame"] > 50:
        tips.append("The biggest object covers most of the frame, so it's probably the background: "
                    "flip `INVERT` in 4a and run 4a → 4c again.")
    tips += ['Set `MODE = "tree"` and look at "Nested outlines". Zero? Then 4b closed every hole: run 4b with `["open"]` only, then 4c again.',
             'Compare "Points stored" for `POINTS = "none"` and `"simple"`. Same shape, fewer points.',
             "`FILL_HOLES = False`: what's left of the objects in stage_4c_contours.png?",
             "Next: `sandbox_4d_boundary.py` peels the crust off these objects."]

    story = (f"👾 I traced every outline in **{src.name}** ({MODE} mode) and found **{len(contours)}**. "
             f"I threw away {dropped} under {MIN_AREA} px² and kept **{n}**"
             + (f". The biggest covers {big['frame']:.1f} % of the frame." if big else ".")
             + (" Everything inside a kept outline is filled in." if FILL_HOLES
                else " Inside the outlines I only kept pixels that were already white."))
    log = write_results("4c", "✂️ Stage 4c · Contour Detection", story,
                        files=[("📥 input", rel(src)), ("📤 output", rel(out)), ("👀 overlay", rel(look))],
                        knobs={"MODE": MODE, "POINTS": POINTS, "MIN_AREA": MIN_AREA, "FILL_HOLES": FILL_HOLES},
                        metrics=metrics, tips=tips, extra="\n".join(table) if rows else "")
    report(metrics, [out, look], log)


def measure(c: np.ndarray, frame_px: int) -> dict:
    """The numbers stage 5 · Extracting will want for one object."""
    area, perimeter = cv2.contourArea(c), cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    m = cv2.moments(c)
    return {"area": area, "frame": 100 * area / frame_px, "box": (x, y, w, h),
            "circ": 4 * np.pi * area / perimeter ** 2 if perimeter else 0.0,
            "solid": area / hull if hull else 0.0,
            "aspect": w / h if h else 0.0,
            "centroid": (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (x + w / 2, y + h / 2)}


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
