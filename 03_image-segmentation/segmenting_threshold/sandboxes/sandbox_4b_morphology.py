"""
✂️ Stage 4b · Morphological Transformation Sandbox
══════════════════════════════════════════════════
One mask in, one repaired mask out, with the same recipe.morphology() CI runs on every frame.

    📥 in   build/sandbox/stage_4a_threshold.png     the mask from 4a
    📤 out  build/sandbox/stage_4b_morphology.png    the repaired mask → the input of 4c
    👀 look build/sandbox/stage_4b_look.png          before | after | changes (orange = sanded away, blue = filled in)
    📝 log  build/sandbox/<unix time>_results_stage_4b.md

Press ▶ in PyCharm (pick "4b · Morphology" in the run menu), or from the repository root:

    python 03_image-segmentation/segmenting_threshold/sandboxes/sandbox_4b_morphology.py
    python 03_image-segmentation/segmenting_threshold/sandboxes/sandbox_4b_morphology.py path/to/any_mask.png

👾 The Robot is a woodworker now. The mask from 4a is a rough plank: splinters (specks) stick out,
   and there are cracks and knotholes (holes) in it. The structuring element is its tool head:
   its shape and size decide how much wood one pass takes off or adds.

   🪚 erode   plane a layer off every edge. Splinters thinner than the tool disappear, the plank shrinks.
   🪵 dilate  spread wood filler along every edge. Cracks narrower than the tool fill up, the plank grows.
   🧽 open    plane, then fill back (erode → dilate). The splinters are gone, the plank keeps its size.
   🔨 close   fill, then plane flush (dilate → erode). The cracks are gone, the plank keeps its size.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "run_pipeline.py").exists())))
from stages import sandbox as sb  # 🔒 the plumbing every sandbox shares (it checks your packages first)

import cv2
import numpy as np

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#    🎛️  TINKER ZONE — change one thing, press ▶, compare the two results files
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
TRY = {}         # empty = exactly what CI does, with the knobs in ../action.yaml. Then try ONE of these:
#   "morphology": ["open", "close"]   run in this order; any of "erode", "dilate", "open", "close"
#   "morph_shape": "rect"             "rect" (flat chisel), "ellipse" (round sanding block), "cross" (plus-shaped)
#   "morph_kernel": 5                 the tool's width in pixels (odd): bigger = more wood per pass
#   "morph_iterations": 2             how many passes per operation
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "segmenting", "4b"
NAMES = ["morphology", "morph_shape", "morph_kernel", "morph_iterations"]   # this sub-step's knobs


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)
    speck = knobs["min_contour_area"]                                        # what 4c will throw away

    # 1. 📥 Load the mask from 4a
    src = sb.pick_input(sb.OUT / "stage_4a_threshold.png", hint="Run sandbox_4a_threshold.py first.")
    before = sb.read_mask(src)

    # 2. 🪚 Work the plank with the recipe, one operation after the other
    mask, ms = sb.timed(recipe.morphology, before, knobs)
    out = sb.save(mask, "stage_4b_morphology.png")

    # 3. 👀 Show what changed: dark = kept, orange = sanded away, blue = filled in (colours are BGR)
    changes = np.full(mask.shape + (3,), 245, np.uint8)
    changes[(before > 0) & (mask > 0)] = (70, 70, 70)
    changes[(before > 0) & (mask == 0)] = (31, 120, 224)     # orange: removed
    changes[(before == 0) & (mask > 0)] = (200, 110, 30)     # blue: added
    steps = " > ".join(knobs["morphology"] or []) or "nothing"
    look = sb.look("stage_4b_look.png", [("before (4a)", before), (f"after: {steps}", mask),
                                         ("changes: orange cut, blue filled", changes)])

    # 4. 📊 Measure before and after
    fg0, fg1 = (100 * np.count_nonzero(m) / m.size for m in (before, mask))
    blobs0, specks0 = count_blobs(before, speck)
    blobs1, specks1 = count_blobs(mask, speck)
    holes0, holes1 = count_holes(before), count_holes(mask)
    changed = 100 * np.count_nonzero(before != mask) / mask.size
    removed = 100 * np.count_nonzero((before > 0) & (mask == 0)) / max(1, np.count_nonzero(before))
    added = 100 * np.count_nonzero((before == 0) & (mask > 0)) / max(1, np.count_nonzero(before))
    drift = 100 * abs(fg1 - fg0) / max(fg0, 1e-9)

    # 5. 🚦 Judge the numbers
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
         f"Blobs under min_contour_area = {speck} px². Lower is better: specks are noise, not objects. Opening removes them."),
        ("🫧", "Blobs", f"{blobs0} → {blobs1}", "good" if blobs1 <= blobs0 else "check",
         "Separate white islands. Lower usually means cleaner, but watch out: closing can also glue two real objects together."),
        ("🕳️", "Holes", f"{holes0} → {holes1}", "good" if holes1 <= holes0 else "check",
         "Black pockets inside white objects. Lower is better for solid objects like the chamber. Closing fills them."),
        ("🪚", "Sanded away", f"{removed:.1f} % of the object", "info",
         "Orange in the look picture. A little is good (splinters); a lot means the tool is too big."),
        ("🪵", "Filled in", f"{added:.1f} % of the object", "info",
         "Blue in the look picture. A little is good (cracks); a lot means objects are growing into each other."),
        ("🔁", "Pixels changed", f"{changed:.2f} % of the image", "info",
         "How much work the Robot did in total. Near 0 % means the operation had nothing to repair."),
        ("⏱️", "Time", f"{ms:.2f} ms ({1000 / max(ms, 1e-3):.0f} fps)", "info",
         "Lower is better. Cost grows with morph_kernel × morph_iterations × number of operations."),
    ]

    tips = []
    if specks1 > 0:
        tips.append('Specks left: put `"open"` first in `"morphology"`, or make `"morph_kernel"` a bit bigger.')
    if holes1 > 0:
        tips.append('Holes left: add `"close"`, or make `"morph_kernel"` a bit bigger.')
    if drift > 25:
        tips.append('The object changed size a lot: use a smaller `"morph_kernel"`, or balance erode and dilate.')
    tips += ['Swap the order: `["close", "open"]` vs `["open", "close"]`. Is the result the same? Why not?',
             'Try `"morph_shape"` rect, ellipse and cross with the same size, and watch the corners in the changes panel.',
             "Next: ▶ `sandbox_4c_contours.py` turns these blobs into objects it can count and measure."]

    k = knobs["morph_kernel"]
    story = (f"👾 I took **{src.name}** and worked it with `recipe.morphology()`: a **{k}×{k} {knobs['morph_shape']}** "
             f"tool, {steps}"
             + (f", {knobs['morph_iterations']} passes each" if (knobs["morph_iterations"] or 1) > 1 else "") + ". "
             f"Specks went from {specks0} to {specks1} and holes from {holes0} to {holes1}, "
             f"while the object changed size by {drift:.0f} %.")
    log = sb.write_results(KEY, STEP, "✂️ Stage 4b · Morphological Transformations", story,
                           files=[("📥 input", sb.rel(src)), ("📤 output", sb.rel(out)), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips)
    sb.report(metrics, [out, look], log, knobs, defaults, NAMES)


def count_blobs(mask: np.ndarray, speck: int) -> tuple[int, int]:
    """(white islands, of which specks)"""
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return len(areas), int((areas < speck).sum())


def count_holes(mask: np.ndarray) -> int:
    """Black pockets fully enclosed by white: background islands that don't touch the image border."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(255 - mask, connectivity=4)
    x, y, w, h = (stats[1:, i] for i in range(4))
    inside = (x > 0) & (y > 0) & (x + w < mask.shape[1]) & (y + h < mask.shape[0])
    return int(inside.sum())


if __name__ == "__main__":
    main()
