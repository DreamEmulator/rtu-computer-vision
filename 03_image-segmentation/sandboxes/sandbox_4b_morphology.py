"""
✂️ Stage 4b · Morphological Transformation Sandbox
══════════════════════════════════════════════════
One image in, one image out.

    📥 in   build/sandbox/stage_4a_threshold.png     the mask from 4a
    📤 out  build/sandbox/stage_4b_morphology.png    the repaired mask → the input of 4c
    👀 look build/sandbox/stage_4b_changes.png       orange = sanded away, blue = filled in
    📝 log  build/sandbox/<unix time>_results_stage_4b.md

Run it from the repository root (or press ▶ in PyCharm):

    python 03_image-segmentation/sandboxes/sandbox_4b_morphology.py
    python 03_image-segmentation/sandboxes/sandbox_4b_morphology.py path/to/any_mask.png

👾 The Robot is a woodworker now. The mask from 4a is a rough plank: splinters (specks) stick out,
   and there are cracks and knotholes (holes) in it. The structuring element is its tool head:
   its shape and size decide how much wood one pass takes off or adds.

   🪚 erode   plane a layer off every edge. Splinters thinner than the tool disappear, the plank shrinks.
   🪵 dilate  spread wood filler along every edge. Cracks narrower than the tool fill up, the plank grows.
   🧽 open    plane, then fill back (erode → dilate). The splinters are gone, the plank keeps its size.
   🔨 close   fill, then plane flush (dilate → erode). The cracks are gone, the plank keeps its size.
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
OPERATIONS = ["open", "close"]   # run in this order; any of "erode", "dilate", "open", "close"
KERNEL_SHAPE = "ellipse"         # "rect" (flat chisel), "ellipse" (round sanding block), "cross" (plus-shaped)
KERNEL_SIZE = 5                  # the tool's width in pixels (odd number): bigger = more wood per pass
ITERATIONS = 1                   # how many passes per operation
SPECK_AREA = 20                  # blobs smaller than this many pixels count as specks (same as 4a)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOLS = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}


def main() -> None:
    # 1. 📥 Load the mask from 4a
    src = pick_input(OUT / "stage_4a_threshold.png", hint="Run sandbox_4a_threshold.py first.")
    before = read_mask(src)

    # 2. 🧰 Pick the tool: the structuring element (kernel)
    if KERNEL_SHAPE not in TOOLS:
        raise SystemExit(f'👾 I don\'t have a "{KERNEL_SHAPE}" tool. Pick rect, ellipse or cross.')
    kernel = cv2.getStructuringElement(TOOLS[KERNEL_SHAPE], (KERNEL_SIZE, KERNEL_SIZE))

    # 3. 🪚 Work the plank, one operation after the other
    mask = before.copy()
    t0 = time.perf_counter()
    for op in OPERATIONS:
        if op == "erode":
            mask = cv2.erode(mask, kernel, iterations=ITERATIONS)                          # 🪚 plane
        elif op == "dilate":
            mask = cv2.dilate(mask, kernel, iterations=ITERATIONS)                         # 🪵 fill
        elif op == "open":
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=ITERATIONS)   # 🧽 plane → fill
        elif op == "close":
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=ITERATIONS)  # 🔨 fill → plane
        else:
            raise SystemExit(f'👾 I don\'t know the operation "{op}". Use erode, dilate, open or close.')
    ms = 1000 * (time.perf_counter() - t0)
    out = save(mask, "stage_4b_morphology.png")

    # 4. 👀 Show what changed: white = kept, orange = sanded away, blue = filled in (colours are BGR)
    changes = np.full(mask.shape + (3,), 245, np.uint8)
    changes[(before > 0) & (mask > 0)] = (70, 70, 70)
    changes[(before > 0) & (mask == 0)] = (31, 120, 224)     # orange: removed
    changes[(before == 0) & (mask > 0)] = (200, 110, 30)     # blue: added
    look = save(changes, "stage_4b_changes.png")

    # 5. 📊 Measure before and after
    fg0, fg1 = (100 * np.count_nonzero(m) / m.size for m in (before, mask))
    blobs0, specks0 = count_blobs(before)
    blobs1, specks1 = count_blobs(mask)
    holes0, holes1 = count_holes(before), count_holes(mask)
    changed = 100 * np.count_nonzero(before != mask) / mask.size
    removed = 100 * np.count_nonzero((before > 0) & (mask == 0)) / max(1, np.count_nonzero(before))
    added = 100 * np.count_nonzero((before == 0) & (mask > 0)) / max(1, np.count_nonzero(before))
    drift = 100 * abs(fg1 - fg0) / max(fg0, 1e-9)

    # 6. 🚦 Judge the numbers
    if fg1 < 0.5:
        fg_level, fg_why = "bad", "The Robot planed the whole plank away. Use a smaller tool or fewer passes."
    elif drift > 25:
        fg_level, fg_why = "check", (f"The object changed size by {drift:.0f} %. Repairs shouldn't change the size much: "
                                     "a smaller tool, or pair every erode with a dilate.")
    else:
        fg_level, fg_why = "good", f"The object changed size by only {drift:.0f} %: repaired, not reshaped."

    metrics = [
        ("⚪", "Foreground", f"{fg0:.1f} % → {fg1:.1f} %", fg_level, fg_why),
        ("🧂", "Specks", f"{specks0} → {specks1}", "good" if specks1 <= specks0 else "check",
         f"Blobs under {SPECK_AREA} px. Lower is better: specks are noise, not objects. Opening removes them."),
        ("🫧", "Blobs", f"{blobs0} → {blobs1}", "good" if blobs1 <= blobs0 else "check",
         "Separate white islands. Lower usually means cleaner, but watch out: closing can also glue two real objects together."),
        ("🕳️", "Holes", f"{holes0} → {holes1}", "good" if holes1 <= holes0 else "check",
         "Black pockets inside white objects. Lower is better for solid objects like the chamber. Closing fills them."),
        ("🪚", "Sanded away", f"{removed:.1f} % of the object", "info",
         "Orange in stage_4b_changes.png. A little is good (splinters); a lot means the tool is too big."),
        ("🪵", "Filled in", f"{added:.1f} % of the object", "info",
         "Blue in stage_4b_changes.png. A little is good (cracks); a lot means objects are growing into each other."),
        ("🔁", "Pixels changed", f"{changed:.2f} % of the image", "info",
         "How much work the Robot did in total. Near 0 % means the operation had nothing to repair."),
        ("⏱️", "Time", f"{ms:.2f} ms ({1000 / max(ms, 1e-3):.0f} fps)", "info",
         "Lower is better. Cost grows with KERNEL_SIZE × ITERATIONS × number of operations."),
    ]

    tips = []
    if specks1 > 0:
        tips.append('Specks left: put `"open"` first, or make `KERNEL_SIZE` a bit bigger.')
    if holes1 > 0:
        tips.append('Holes left: add `"close"`, or make `KERNEL_SIZE` a bit bigger.')
    if drift > 25:
        tips.append("The object changed size a lot: use a smaller `KERNEL_SIZE`, or balance erode and dilate.")
    tips += ['Swap the order: `["close", "open"]` vs `["open", "close"]`. Is the result the same? Why not?',
             "Try `KERNEL_SHAPE` rect, ellipse and cross with the same size, and watch the corners in stage_4b_changes.png.",
             "Next: `sandbox_4c_contours.py` turns these blobs into objects it can count and measure."]

    story = (f"👾 I took **{src.name}** and worked it with a **{KERNEL_SIZE}×{KERNEL_SIZE} {KERNEL_SHAPE}** tool: "
             f"{' → '.join(OPERATIONS) or 'nothing'}"
             + (f", {ITERATIONS} passes each" if ITERATIONS > 1 else "") + ". "
             f"Specks went from {specks0} to {specks1} and holes from {holes0} to {holes1}, "
             f"while the object changed size by {drift:.0f} %.")
    log = write_results("4b", "✂️ Stage 4b · Morphological Transformations", story,
                        files=[("📥 input", rel(src)), ("📤 output", rel(out)), ("👀 changes", rel(look))],
                        knobs={"OPERATIONS": OPERATIONS, "KERNEL_SHAPE": KERNEL_SHAPE,
                               "KERNEL_SIZE": KERNEL_SIZE, "ITERATIONS": ITERATIONS},
                        metrics=metrics, tips=tips)
    report(metrics, [out, look], log)


def count_blobs(mask: np.ndarray) -> tuple[int, int]:
    """(white islands, of which specks)"""
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return len(areas), int((areas < SPECK_AREA).sum())


def count_holes(mask: np.ndarray) -> int:
    """Black pockets fully enclosed by white: background islands that don't touch the image border."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(255 - mask, connectivity=4)
    x, y, w, h = (stats[1:, i] for i in range(4))
    inside = (x > 0) & (y > 0) & (x + w < mask.shape[1]) & (y + h < mask.shape[0])
    return int(inside.sum())


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
