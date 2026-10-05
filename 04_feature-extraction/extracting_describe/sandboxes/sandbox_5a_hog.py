"""
🧬 Stage 5a · Histogram of Oriented Gradients Sandbox
═════════════════════════════════════════════════════
One real frame in, one HOG vector out, with the same recipe.hog_features() CI runs on every frame.

    📥 in   your snap, exactly as stage 4 · Segmenting passed it on (build/segmenting/)
    👀 look build/sandbox/stage_5a_look.png        the frame | its edges | HOG's stars | one cell's histogram
    📝 log  build/sandbox/<unix time>_results_stage_5a.md

Press ▶ in PyCharm (pick "5a · HOG" in the run menu), or from the repository root:

    python 04_feature-extraction/extracting_describe/sandboxes/sandbox_5a_hog.py
    python 04_feature-extraction/extracting_describe/sandboxes/sandbox_5a_hog.py low_fluid

👾 The Robot cuts the frame into cells and asks, in every cell: which way do the edges point, and
   how strongly? It measures the slope of the brightness at every pixel (Sobel's derivatives), files
   each pixel under one of a few directions, weighted by how steep it is, and normalises groups of
   cells (blocks) so a dim frame and a bright frame of the same shape give the same numbers
   (Dalal & Triggs, 2005: they built it to find pedestrians). Shape, as a list of numbers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "run_pipeline.py").exists())))
from stages import sandbox as sb  # 🔒 the plumbing every sandbox shares (it checks your packages first)

import cv2
import numpy as np
from skimage.feature import hog

from stages.s5_extracting import hog_picture  # 🔒 the same stars CI draws in its preview

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#    🎛️  TINKER ZONE — change one thing, press ▶, compare the two results files
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
FRAME = "snap"   # "snap" = your Step 0 photo · "drop", "no_drop", "low_fluid" = a test frame of that class
TRY = {}         # empty = exactly what CI does, with the knobs in ../action.yaml. Then try ONE of these:
#   "hog_pixels_per_cell": 8     smaller cells: finer detail, and watch the vector length explode
#   "hog_orientations": 4        only 4 directions (every 45°) instead of 9 (every 20°)
#   "hog_cells_per_block": 1     normalise every cell on its own instead of in 2 × 2 groups
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "extracting", "5a"
NAMES = ["hog_pixels_per_cell", "hog_orientations", "hog_cells_per_block"]   # this sub-step's knobs


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)

    # 1. 📥 Load the frame stage 4 · Segmenting passed on
    img, space, src = sb.frame("segmenting", FRAME)
    gray = sb.to_luma(img, space)

    # 2. 🧭 HOG: the recipe turns the frame into numbers
    vec, ms = sb.timed(recipe.hog_features, gray, knobs)
    p = recipe.hog_params(knobs)
    ppc, cpb, o = p["pixels_per_cell"][0], p["cells_per_block"][0], p["orientations"]
    ncy, ncx = gray.shape[0] // ppc, gray.shape[1] // ppc
    nby, nbx = ncy - cpb + 1, ncx - cpb + 1

    # 3. 🔍 What HOG measures: the slope of the brightness (Sobel), and the cell with the steepest slopes
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = np.hypot(gx, gy)
    per_cell = magnitude[:ncy * ppc, :ncx * ppc].reshape(ncy, ppc, ncx, ppc).sum(axis=(1, 3))
    cy, cx = np.unravel_index(np.argmax(per_cell), per_cell.shape)
    blocks = hog(gray, feature_vector=False, **p)                       # block row, block col, cell row, cell col, direction
    by, bx = min(cy, nby - 1), min(cx, nbx - 1)
    hist = blocks[by, bx, cy - by, cx - bx]                              # that cell's directions, as HOG stores them
    top = int(np.argmax(hist))
    edges_share = 100 * float((magnitude > 0.25 * magnitude.max()).mean()) if magnitude.max() > 0 else 0.0

    # 4. 👀 The frame, its edges, HOG's stars, and the histogram of the strongest cell
    s = sb.zoom(gray.shape)
    frame_vis = sb.bigger(gray)
    for y in range(0, ncy * ppc, ppc):                                   # the cell grid
        cv2.line(frame_vis, (0, y * s), (ncx * ppc * s, y * s), (90, 90, 90), 1)
    for x in range(0, ncx * ppc, ppc):
        cv2.line(frame_vis, (x * s, 0), (x * s, ncy * ppc * s), (90, 90, 90), 1)
    cv2.rectangle(frame_vis, (cx * ppc * s, cy * ppc * s), ((cx + 1) * ppc * s - 1, (cy + 1) * ppc * s - 1), (31, 140, 240), 3)
    edges = np.uint8(255 * magnitude / max(magnitude.max(), 1e-9))
    fig = sb.new_figure(3.8, 3.8)
    ax = fig.add_subplot(projection="polar")
    angles = np.deg2rad((np.arange(o) + 0.5) * 180 / o)
    width = np.deg2rad(180 / o) * 0.9
    ax.bar(np.concatenate([angles, angles + np.pi]), np.concatenate([hist, hist]), width=width,
           color=["#e65100" if i % o == top else "#90a4ae" for i in range(2 * o)])
    ax.set_yticklabels([]); ax.set_title(f"cell ({cy},{cx}): {o} directions", fontsize=10)
    fig.tight_layout()
    look = sb.look("stage_5a_look.png", [("frame + cells (orange: strongest)", frame_vis), ("edges (Sobel)", edges),
                                         ("HOG stars", hog_picture(gray, knobs)), ("strongest cell", sb.figure(fig))])

    # 5. 🚦 Judge the numbers
    gate = knobs["gate_max_features"]
    lo = int(top * 180 / o)
    metrics = [
        ("📏", "HOG vector length", f"{len(vec)}", "good" if gate is None or len(vec) <= gate else "bad",
         f"{nby} × {nby if nby == nbx else nbx} blocks × {cpb}×{cpb} cells × {o} directions. "
         f"The 🚦 gate allows {gate} numbers for the whole vector (the camera chip's memory), HOG + LBP + …"),
        ("🔲", "Cells", f"{ncy} × {ncx} of {ppc}×{ppc} px", "info",
         "One direction histogram per cell. A cell bigger than the drop can't tell you much about the drop."),
        ("🧭", "Strongest cell", f"({cy},{cx}), mostly {lo}–{lo + 180 // o}°", "info",
         "Direction of the brightness slope: 0° = brightness changes left↔right, so a vertical edge (the chamber's "
         "walls). 90° = it changes top↔bottom: a horizontal edge (the fluid line)."),
        ("✏️", "Edge pixels", f"{edges_share:.1f} %", "info",
         "Pixels with a slope of at least a quarter of the steepest. HOG only cares about these; flat areas add nothing."),
        ("🗜️", "Pixels in → numbers out", f"{gray.size} → {len(vec)}", "info",
         "A description should be shorter than what it describes, and keep what matters."),
        ("⏱️", "Time", f"{ms:.2f} ms ({1000 / max(ms, 1e-3):.0f} fps)", "info",
         "One frame. Smaller cells mean more histograms and more time."),
    ]
    tips = ["Is the orange cell on the drip chamber, or on clutter stage 4 let through? HOG describes whatever "
            "has edges: if that's the background, go back to the 4a–4d sandboxes.",
            '`"hog_pixels_per_cell": 8`: four times as many cells. Watch the vector length and the gate.',
            '`"hog_orientations": 4`: can it still tell the fluid line (horizontal) from the walls (vertical)?',
            'Compare classes: `FRAME = "drop"`, then `"low_fluid"`. Where do the stars differ?',
            "Next: ▶ `sandbox_5b_lbp.py`: not which way the edges point, but what the texture looks like."]

    story = (f"👾 I took **{src.name}** as stage 4 passed it on ({gray.shape[1]}×{gray.shape[0]} px), cut it into "
             f"{ncy}×{ncx} cells of {ppc}×{ppc} px and asked every cell which way its edges point, in {o} directions, "
             f"with `recipe.hog_features()`. That's **{len(vec)} numbers**. The steepest cell is ({cy},{cx}), "
             f"with its edges mostly at {lo}–{lo + 180 // o}°.")
    log = sb.write_results(KEY, STEP, "🧬 Stage 5a · Histogram of Oriented Gradients", story,
                           files=[("📥 input", sb.rel(src)), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips)
    sb.report(metrics, [look], log, knobs, defaults, NAMES)


if __name__ == "__main__":
    main()
