"""
✂️ Stage 4c · Contour Detection Sandbox
═══════════════════════════════════════
One mask in, the objects we keep out, with the same recipe.keep_contours() CI runs on every frame.

    📥 in   build/sandbox/stage_4b_morphology.png   the repaired mask from 4b
    📤 out  build/sandbox/stage_4c_contours.png     only the objects we keep → the input of 4d
    👀 look build/sandbox/stage_4c_look.png         every kept contour on the photo | the objects
    📝 log  build/sandbox/<unix time>_results_stage_4c.md

Press ▶ in PyCharm (pick "4c · Contours" in the run menu), or from the repository root:

    python 03_image-segmentation/segmenting_threshold/sandboxes/sandbox_4c_contours.py
    python 03_image-segmentation/segmenting_threshold/sandboxes/sandbox_4c_contours.py path/to/any_mask.png

👾 Tracing your hand: put your hand on paper and follow its outline without lifting the pencil.
   The Robot does the same for every white blob (Suzuki & Abe, 1985: follow the border pixel by pixel
   until you're back where you started). Instead of "a block of white pixels" it now has a list of
   (x, y) points around each object, so it can count objects and measure them.
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
#   "contour_mode": "tree"      "external" only the outer outline of each object
#                               "list"     every outline, holes too, no family tree
#                               "ccomp"    two levels: outer outlines and their holes
#                               "tree"     every outline with the full family tree (the Kinder Surprise)
#   "contour_points": "none"    "none" store every border pixel · "simple" store only the corners of straight runs
#   "min_contour_area": 40      outlines enclosing fewer pixels than this are specks: throw them away
#   "fill_holes": False         True: colour in everything inside a kept outline (Jordan, 1887: a closed curve has an inside)
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "segmenting", "4c"
NAMES = ["contour_mode", "contour_points", "min_contour_area", "fill_holes"]   # this sub-step's knobs


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)
    min_area = knobs["min_contour_area"]

    # 1. 📥 Load the repaired mask from 4b
    src = sb.pick_input(sb.OUT / "stage_4b_morphology.png", hint="Run sandbox_4b_morphology.py first.")
    mask = sb.read_mask(src)

    # 2. ✏️ Trace every outline, 🧹 throw away the specks, 🎨 paint the objects we keep: the recipe does all three
    (objects, kept), ms = sb.timed(recipe.keep_contours, mask, knobs)
    out = sb.save(objects, "stage_4c_contours.png")
    contours, hierarchy = recipe.find_contours(mask, knobs)                  # every outline, to count what was dropped

    # 3. 📏 Measure every kept object, biggest first
    kept = sorted(kept, key=cv2.contourArea, reverse=True)
    rows = [measure(c, mask.size) for c in kept]
    points = sum(len(c) for c in contours)
    children = int((hierarchy[0][:, 3] != -1).sum()) if hierarchy is not None else 0
    dropped = len(contours) - len(kept)

    # 4. 👀 Draw on the photo from 4a (or on the mask, if the photo doesn't match), enlarged so thin lines stay thin
    photo = sb.read_photo("stage_4_photo.png", mask.shape)
    vis = sb.bigger(photo if photo is not None else (mask // 3 + 40).astype(np.uint8))
    s = sb.zoom(mask.shape)
    big = [(c * s + s // 2).astype(np.int32) for c in kept]
    cv2.drawContours(vis, big, -1, (31, 140, 240), 2)                         # orange: every kept object
    if kept:
        x, y, w, h = cv2.boundingRect(big[0])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (200, 110, 30), 2)         # blue: the biggest one's box
        cv2.drawContours(vis, [cv2.convexHull(big[0])], -1, (90, 200, 90), 1)  # green: its rubber band (hull)
        cx, cy = rows[0]["centroid"]
        cv2.circle(vis, (round(cx * s + s / 2), round(cy * s + s / 2)), 5, (90, 200, 90), -1)  # green dot: its centre
    look = sb.look("stage_4c_look.png", [(f"{len(kept)} kept contours", vis), ("objects -> 4d", objects)])

    # 5. 🚦 Judge the numbers
    n = len(kept)
    if n == 0:
        n_level, n_why = "bad", "Nothing left to measure. Lower min_contour_area, or go back to 4a/4b."
    elif n <= 10:
        n_level, n_why = "good", "A handful of clear objects: easy to pick the drip chamber from."
    else:
        n_level, n_why = "check", "More than 10 objects: for a drip frame that's probably noise. Raise min_contour_area, or clean harder in 4b."
    big = rows[0] if rows else None
    metrics = [
        ("✏️", "Outlines traced", f"{len(contours)}", "info",
         f"Every outline the Robot found with contour_mode = {knobs['contour_mode']} (holes count as outlines in list, ccomp and tree)."),
        ("🧹", "Thrown away", f"{dropped} ({100 * dropped / max(1, len(contours)):.0f} %)", "info",
         f"Outlines enclosing less than {min_area} px². A high share means 4b left a lot of salt."),
        ("📦", "Objects kept", f"{n}", n_level, n_why),
        ("🪆", "Nested outlines", f"{children}", "info",
         "Outlines inside another outline (holes, or objects in holes). Always 0 with contour_mode = external."),
        ("📍", "Points stored", f"{points}", "info",
         'Lower is cheaper. contour_points = "simple" keeps only corners of straight runs; "none" keeps every border pixel.'),
    ]
    if big:
        frame_level = ("bad" if big["frame"] > 90 else "check" if big["frame"] > 50 or big["frame"] < 0.5
                       else "good")
        metrics += [
            ("🐘", "Biggest object", f"{big['area']:.0f} px² ({big['frame']:.1f} % of the frame)", frame_level,
             "Over half the frame is suspicious and over 90 % is almost certainly the background: "
             "flip invert in 4a. Under 0.5 % means even the biggest object is tiny."),
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
        tips.append('Raise `"min_contour_area"` until only real objects are left. How many pixels is the drip chamber?')
    if big and big["frame"] > 50:
        tips.append("The biggest object covers most of the frame, so it's probably the background: "
                    'flip `"invert"` in 4a and run 4a → 4c again.')
    tips += ['Set `"contour_mode": "tree"` and look at "Nested outlines". Zero? Then 4b closed every hole: run 4b with `["open"]` only, then 4c again.',
             'Compare "Points stored" for `"contour_points"` `"none"` and `"simple"`. Same shape, fewer points.',
             '`"fill_holes": False`: what\'s left of the objects in the look picture?',
             "Next: ▶ `sandbox_4d_boundary.py` peels the crust off these objects."]

    story = (f"👾 I traced every outline in **{src.name}** with `recipe.keep_contours()` ({knobs['contour_mode']} mode) "
             f"and found **{len(contours)}**. I threw away {dropped} under {min_area} px² and kept **{n}**"
             + (f". The biggest covers {big['frame']:.1f} % of the frame." if big else ".")
             + (" Everything inside a kept outline is filled in." if knobs["fill_holes"]
                else " Inside the outlines I only kept pixels that were already white."))
    log = sb.write_results(KEY, STEP, "✂️ Stage 4c · Contour Detection", story,
                           files=[("📥 input", sb.rel(src)), ("📤 output", sb.rel(out)), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips,
                           extra="\n".join(table) if rows else "")
    sb.report(metrics, [out, look], log, knobs, defaults, NAMES)


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


if __name__ == "__main__":
    main()
