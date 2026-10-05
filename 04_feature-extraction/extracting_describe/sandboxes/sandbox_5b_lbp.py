"""
🧬 Stage 5b · Local Binary Patterns Sandbox
═══════════════════════════════════════════
One real frame in, one LBP vector out, with the same recipe.lbp_codes() and lbp_features() CI runs on every frame.

    📥 in   your snap, exactly as stage 4 · Segmenting passed it on (build/segmenting/)
    👀 look build/sandbox/stage_5b_look.png        the frame + grid | a texture code per pixel | the codes counted
    📝 log  build/sandbox/<unix time>_results_stage_5b.md

Press ▶ in PyCharm (pick "5b · LBP" in the run menu), or from the repository root:

    python 04_feature-extraction/extracting_describe/sandboxes/sandbox_5b_lbp.py
    python 04_feature-extraction/extracting_describe/sandboxes/sandbox_5b_lbp.py drop

👾 Every pixel looks at P neighbours on a circle around it and writes down a 1 for every neighbour at
   least as bright as itself, a 0 for every darker one. Eight neighbours, eight bits: one byte per pixel
   that says "spot", "edge", "corner", "flat" or "noise" (Ojala, Pietikäinen & Harwood, University of
   Oulu, 1994–96). Counting those codes per cell of a grid gives texture, and roughly where it is.
   HOG (5a) says which way the edges point; LBP says what the surface feels like.
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
FRAME = "snap"   # "snap" = your Step 0 photo · "drop", "no_drop", "low_fluid" = a test frame of that class
TRY = {}         # empty = exactly what CI does, with the knobs in ../action.yaml. Then try ONE of these:
#   "lbp_radius": 3       neighbours 3 px away: coarser texture, small noise matters less
#   "lbp_points": 16      16 neighbours on the circle (pair it with radius 2 or more): finer, but 18 codes per cell
#   "lbp_grid": 1         one histogram for the whole frame: WHAT textures, but no longer WHERE
#   "lbp_grid": 8         64 cells: more WHERE, and 4 times as many numbers
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "extracting", "5b"
NAMES = ["lbp_points", "lbp_radius", "lbp_grid"]   # this sub-step's knobs


def kind(code: int, p: int) -> str:
    return {0: "bright spot", p: "flat / dark spot", p + 1: "noisy"}.get(code, f"edge {code}/{p}")


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)
    p, grid = knobs["lbp_points"], knobs["lbp_grid"]

    # 1. 📥 Load the frame stage 4 · Segmenting passed on
    img, space, src = sb.frame("segmenting", FRAME)
    gray = sb.to_luma(img, space)

    # 2. 🔢 A code per pixel, then a histogram of codes per grid cell: both from the recipe
    codes = recipe.lbp_codes(gray, knobs).astype(int)
    vec, ms = sb.timed(recipe.lbp_features, gray, knobs)
    counts = np.bincount(codes.ravel(), minlength=p + 2) / codes.size
    share = {"flat": counts[p], "edges": counts[1:p].sum(), "spots": counts[0], "noisy": counts[p + 1]}

    # 3. 👀 The frame with its grid, the codes as colours, and the codes counted
    s = sb.zoom(gray.shape)
    frame_vis = sb.bigger(gray)
    h, w = gray.shape
    for g in range(1, grid):
        cv2.line(frame_vis, (0, g * h // grid * s), (w * s, g * h // grid * s), (31, 140, 240), 2)
        cv2.line(frame_vis, (g * w // grid * s, 0), (g * w // grid * s, h * s), (31, 140, 240), 2)
    palette = np.zeros((p + 2, 3), np.uint8)                              # BGR per code
    palette[0] = (60, 220, 255)                                           # bright spot: yellow
    palette[p] = (40, 40, 40)                                             # flat / dark spot: near black
    palette[p + 1] = (60, 60, 230)                                        # noisy: red
    for c in range(1, p):                                                 # edges: green → blue as more neighbours are brighter
        palette[c] = (int(80 + 150 * c / p), int(200 - 100 * c / p), 60)
    code_vis = cv2.resize(palette[codes], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    fig = sb.new_figure(5, 3.8)
    ax = fig.add_subplot()
    ax.bar(range(p + 2), counts, color=[tuple(palette[c][::-1] / 255) for c in range(p + 2)], edgecolor="#555")
    ax.set_xticks(range(p + 2), ["spot"] + [str(c) for c in range(1, p)] + ["flat", "noisy"], fontsize=8)
    rest = np.delete(counts, p).max()
    if counts[p] > 2 * rest > 0:                                          # one huge "flat" bar would hide the rest
        ax.set_ylim(0, 1.5 * rest)
        ax.text(p, 1.42 * rest, f"{100 * counts[p]:.0f} % ↑", ha="center", va="top", fontsize=9, color="white",
                bbox={"facecolor": "#333", "edgecolor": "none", "pad": 2})
    ax.set_ylabel("share of pixels"); ax.set_xlabel("LBP code (1 … P−1: edges)")
    ax.set_title(f"the whole frame · CI keeps one of these per cell ({grid}×{grid})", fontsize=9)
    fig.tight_layout()
    look = sb.look("stage_5b_look.png", [(f"frame + {grid}x{grid} grid", frame_vis), ("a code per pixel", code_vis),
                                         ("the codes counted", sb.figure(fig))])

    # 4. 🚦 Judge the numbers
    metrics = [
        ("📏", "LBP vector length", f"{len(vec)}", "info",
         f"{grid}×{grid} cells × {p + 2} codes. Short next to HOG: LBP is cheap texture."),
        ("⬛", "Flat", f"{100 * share['flat']:.0f} %", "check" if share["flat"] > 0.7 else "info",
         "Pixels whose neighbours are all at least as bright: flat areas, like the black background stage 4 masked "
         "away. Lots of flat = lots of numbers that say 'nothing here'."),
        ("〰️", "Edges and corners", f"{100 * share['edges']:.0f} %", "info",
         "Codes 1 … P−1: some neighbours brighter, some darker, in one unbroken run. The outline of the chamber, the drop."),
        ("✨", "Spots", f"{100 * share['spots']:.1f} %", "info", "Code 0: every neighbour darker. A tiny bright speck."),
        ("🌫️", "Noisy", f"{100 * share['noisy']:.0f} %", "check" if share["noisy"] > 0.3 else "good",
         "No clear pattern: brighter and darker neighbours mixed up. Sensor noise lands here; stage 2 · Cleaning should keep it low."),
        ("⏱️", "Time", f"{ms:.2f} ms ({1000 / max(ms, 1e-3):.0f} fps)", "info",
         "8 comparisons per pixel, then counting. Cheap enough for a camera chip."),
    ]
    tips = []
    if share["noisy"] > 0.3:
        tips.append("Lots of noisy codes: stage 2 · Cleaning could smooth more, or try `\"lbp_radius\": 2`.")
    tips += ['`"lbp_grid": 1`: the same codes, but the frame no longer knows WHERE they are. Does that matter for a drop?',
             'Compare classes: `FRAME = "drop"`, then `"low_fluid"`. Which codes change?',
             "Next: ▶ `sandbox_5c_scaling.py`: HOG and LBP numbers live on different rulers. Put them on one."]

    story = (f"👾 I gave every pixel of **{src.name}** a texture code with `recipe.lbp_codes()`: {p} neighbours on a "
             f"circle of radius {knobs['lbp_radius']}. {100 * share['flat']:.0f} % came out flat, "
             f"{100 * share['edges']:.0f} % edges or corners and {100 * share['noisy']:.0f} % noisy. "
             f"Counted per cell of a {grid}×{grid} grid, that's **{len(vec)} numbers**.")
    log = sb.write_results(KEY, STEP, "🧬 Stage 5b · Local Binary Patterns", story,
                           files=[("📥 input", sb.rel(src)), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips)
    sb.report(metrics, [look], log, knobs, defaults, NAMES)


if __name__ == "__main__":
    main()
