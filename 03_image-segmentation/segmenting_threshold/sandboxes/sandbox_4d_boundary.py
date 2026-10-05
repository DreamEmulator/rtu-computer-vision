"""
✂️ Stage 4d · Boundary Extraction Sandbox
═════════════════════════════════════════
The objects in, their crust out, with the same recipe.boundary() and recipe.apply() CI runs on every frame.

    📥 in   build/sandbox/stage_4c_contours.png    the objects from 4c
    📤 out  build/sandbox/stage_4d_boundary.png    the mask CI ends up with
    👀 look build/sandbox/stage_4d_look.png        the crust on the photo | what stage 5 · Extracting receives
    📝 log  build/sandbox/<unix time>_results_stage_4d.md

Press ▶ in PyCharm (pick "4d · Boundary" in the run menu), or from the repository root:

    python 03_image-segmentation/segmenting_threshold/sandboxes/sandbox_4d_boundary.py
    python 03_image-segmentation/segmenting_threshold/sandboxes/sandbox_4d_boundary.py path/to/any_mask.png

👾 The cookie crust. Take a solid baked cookie and scoop out the soft centre: what's left is the crust.
   The Robot does exactly that with 4b's tools. It makes a slightly eroded (shrunk) copy of the object
   and subtracts it from the original: β(A) = A − (A ⊖ B). Only the crust is left.

   Contours (4c) give you the outline as a list of (x, y) points, which is good for measuring.
   Boundaries give you the outline as pixels in an image, which is good for drawing and overlays.
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
TRY = {"boundary": "inner"}   # CI passes the whole objects on (boundary: none), so this sandbox starts
#                               with a crust. Set TRY = {} to see exactly what CI does. Then try ONE of these:
#   "boundary": "outer"       "none"     no crust: the whole objects go on to stage 5
#                             "inner"    the crust inside the cookie:  A − erode(A)
#                             "outer"    a ring of icing just outside:  dilate(A) − A
#                             "gradient" both at once, a thicker line:   dilate(A) − erode(A)
#   "boundary_shape": "ellipse"   "rect", "ellipse" or "cross": the shape of the scoop
#   "boundary_kernel": 5          odd: for inner and outer, 3 gives a crust about 1 px thick, 5 about 2 px
#   "pass_on": "mask"             what stage 5 receives: "masked" | "mask" | "crop" | "original"
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "segmenting", "4d"
NAMES = ["boundary", "boundary_shape", "boundary_kernel", "pass_on"]   # this sub-step's knobs


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)

    # 1. 📥 Load the objects from 4c
    src = sb.pick_input(sb.OUT / "stage_4c_contours.png", hint="Run sandbox_4c_contours.py first.")
    mask = sb.read_mask(src)
    if not mask.any():
        raise SystemExit("👾 The input has no white pixels, so there's no cookie to take the crust from. Check 4a–4c.")

    # 2. 🍪 Scoop out the centre, keep the crust
    crust, ms = sb.timed(recipe.boundary, mask, knobs)
    out = sb.save(crust, "stage_4d_boundary.png")

    # 3. 📦 What stage 5 receives: the recipe's apply(), on the photo from 4a
    photo = sb.read_photo("stage_4_photo.png", mask.shape)
    base = photo if photo is not None else (mask // 3 + 40).astype(np.uint8)
    contours, _ = recipe.find_contours(mask, knobs)
    passed = recipe.apply(base, crust, contours, knobs["pass_on"])

    # 4. 👀 Paint the crust on the photo
    vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    vis[crust > 0] = (31, 140, 240)                                          # orange crust (BGR)
    look = sb.look("stage_4d_look.png", [(f"boundary: {knobs['boundary']}", vis),
                                         (f"pass_on: {knobs['pass_on']} -> stage 5", passed)])

    # 5. 📊 Measure the crust
    object_px, crust_px = np.count_nonzero(mask), np.count_nonzero(crust)
    outlines, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)   # outer outlines + holes
    perimeter = sum(cv2.arcLength(c, True) for c in outlines)
    thickness = crust_px / perimeter if perimeter else 0.0
    share = 100 * crust_px / max(1, object_px)
    pieces = cv2.connectedComponents(crust, connectivity=8)[0] - 1
    expected = len(outlines)

    # 6. 🚦 Judge the numbers
    if knobs["boundary"] == "none":
        piece_level, piece_why = "info", "boundary is none: the whole objects go on, there is no crust to break."
    elif pieces == expected:
        piece_level, piece_why = "good", "One closed crust per outline: every object and hole got a complete border."
    elif pieces > expected:
        piece_level, piece_why = "check", ("More pieces than outlines: some crusts broke. Either an object is thinner "
                                           "than the scoop (try a smaller boundary_kernel), or it touches the edge of "
                                           "the image, where OpenCV leaves the crust open.")
    else:
        piece_level, piece_why = "check", ("Fewer pieces than outlines: crusts touch and merge, where objects or "
                                           "holes lie closer together than the crust is thick.")
    metrics = [
        ("🍪", "Object pixels", f"{object_px}", "info", "The whole cookie: every white pixel of the input."),
        ("🥧", "Crust pixels", f"{crust_px}", "info", "What's left after scooping. This is the mask CI ends up with."),
        ("📏", "Crust thickness", f"{thickness:.1f} px", "info",
         "Crust pixels ÷ outline length. About 1 px = a crisp line for precise drawing. "
         "Thicker is easier to see and survives small shifts, but blurs detail."),
        ("🫓", "Crust share", f"{share:.1f} % of the object",
         "info" if knobs["boundary"] == "none" else "check" if share > 60 else "good",
         "Low = solid objects with a thin rim (what we want from 4c). Near 100 % = the objects are "
         "already thin lines: was fill_holes off in 4c? (Or boundary is none: then it's always 100 %.)"),
        ("🧩", "Crust pieces", f"{pieces} (outlines: {expected})", piece_level, piece_why),
        ("⏱️", "Time", f"{ms:.2f} ms ({1000 / max(ms, 1e-3):.0f} fps)", "info",
         "Lower is better. One erosion or dilation plus a subtraction: cheap."),
    ]

    tips = ['Try `"boundary": "outer"`: the crust now sits outside the object. When would you want that?',
            'Set `"boundary_kernel"` to 3, 5 and 7 and watch the thickness go up by about 1 px per step.',
            "Does a crust help the classifier? Put `boundary: \"inner\"` in ../action.yaml and ▶ `recipe.py`: "
            "it runs every frame through Extracting and the random forest. Compare the test accuracy with `none`.",
            "Next stop: stage 5 · Extracting turns these objects into numbers a classifier can learn from."]

    k = knobs["boundary_kernel"]
    story = (f"👾 I took the objects in **{src.name}** and ran `recipe.boundary()` with **{knobs['boundary']}**"
             + ("" if knobs["boundary"] == "none" else f" and a **{k}×{k} {knobs['boundary_shape']}** scoop") + ". "
             f"**{crust_px}** pixels are left of **{object_px}**"
             + ("" if knobs["boundary"] == "none" else f": a crust about **{thickness:.1f} px** thick in {pieces} "
                f"piece{'s' if pieces != 1 else ''}")
             + f". With `pass_on: {knobs['pass_on']}`, that's what stage 5 · Extracting receives.")
    log = sb.write_results(KEY, STEP, "✂️ Stage 4d · Boundary Extraction", story,
                           files=[("📥 input", sb.rel(src)), ("📤 output", sb.rel(out)), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips)
    sb.report(metrics, [out, look], log, knobs, defaults, NAMES)


if __name__ == "__main__":
    main()
