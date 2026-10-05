"""
✂️ Stage 4a · Thresholding Sandbox
══════════════════════════════════
One real frame in, one mask out, with the same recipe.threshold() CI runs on every frame.

    📥 in   your snap, exactly as stage 3 · Improving passed it on (build/improving/)
    📤 out  build/sandbox/stage_4a_threshold.png   the mask → the input of 4b
    👀 look build/sandbox/stage_4a_look.png        input | mask, enlarged. Keep it open: it refreshes
    📝 log  build/sandbox/<unix time>_results_stage_4a.md

Press ▶ in PyCharm (pick "4a · Threshold" in the run menu), or from the repository root:

    python 03_image-segmentation/segmenting_threshold/sandboxes/sandbox_4a_threshold.py
    python 03_image-segmentation/segmenting_threshold/sandboxes/sandbox_4a_threshold.py path/to/photo.jpg

👾 The Robot sorts every pixel into one of two bins: object (white) or background (black).
   The thresholding algorithm is its brain: the rule that decides which bin a pixel lands in.
   Global, Otsu, Triangle and Adaptive are four different brains.
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
#   "threshold": "otsu"     "global"   one fixed T for every pixel (you pick it with global_value)
#                           "otsu"     tries all 256 T's, keeps the one that best splits TWO hills (Otsu, 1979)
#                           "triangle" made for ONE big hill with a tail, like our grey wall (Zack et al., 1977)
#                           "adaptive" a different T for every neighbourhood: fair under uneven light
#   "global_value": 100     only for global: 0 = black … 255 = white
#   "adaptive_block": 51    only for adaptive: the neighbourhood size in pixels (odd)
#   "adaptive_c": 2         only for adaptive: how much darker than its neighbours a pixel must be
#   "invert": False         True: DARK pixels are the object (the drip chamber is darker than the wall)
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "segmenting", "4a"
NAMES = ["threshold", "global_value", "adaptive_block", "adaptive_c", "invert"]   # this sub-step's knobs


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)

    # 1. 📥 Load the frame stage 3 · Improving passed on, and take its brightness
    img, space, src = sb.frame("improving", FRAME)
    gray = sb.to_luma(img, space)                                            # what s4_segmenting does too
    sb.save(gray, "stage_4_photo.png")                                       # 4c and 4d draw their overlays on this

    # 2. 🧠 Threshold: the recipe decides which bin every pixel lands in
    (mask, T), ms = sb.timed(recipe.threshold, gray, knobs)
    out = sb.save(mask, "stage_4a_threshold.png")
    look = sb.look("stage_4a_look.png", [("input (stage 3)", gray), (f"4a {knobs['threshold']} mask", mask)])

    # 3. 📊 Measure what happened
    speck = knobs["min_contour_area"]                                        # what 4c will throw away
    fg = 100 * np.count_nonzero(mask) / mask.size                            # share of pixels in the object bin
    otsu_T, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    adaptive = np.isnan(T)
    eta = separability(gray, otsu_T if adaptive else T)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]                                       # label 0 is the background
    blobs, specks = len(areas), int((areas < speck).sum())
    biggest = 100 * areas.max() / max(1, areas.sum()) if blobs else 0.0

    # 4. 🚦 Judge the numbers
    if fg < 0.5:
        fg_level, fg_why = "bad", "Almost nobody got in: the rule is too strict. Move T, or flip invert."
    elif fg > 95:
        fg_level, fg_why = "bad", "Everybody got in: the mask is all object. Flip invert, or move T."
    elif fg > 60:
        fg_level, fg_why = "check", "More object than background. For a drip chamber in a frame that's suspicious: is invert right?"
    else:
        fg_level, fg_why = "good", "Some pixels in, most out. The pipeline's gates fail below 0.5 % (empty) and above 95 % (full)."
    eta_level = "good" if eta >= 0.8 else "check" if eta >= 0.5 else "bad"
    speck_share = 100 * specks / max(1, blobs)

    metrics = [
        ("🎯", "Threshold T", "per neighbourhood" if adaptive else f"{T:.0f}", "info",
         "The brightness where the Robot switches bins. "
         + ("Adaptive uses a different T for every pixel's neighbourhood." if adaptive
            else f"For comparison, Otsu would pick {otsu_T:.0f}.")),
        ("⚪", "Foreground", f"{fg:.1f} %", fg_level, fg_why),
        ("⛰️", "Separability η", f"{eta:.2f}" + (" (at Otsu's T)" if adaptive else ""),
         "info" if adaptive else eta_level,
         "0 → 1, higher is better: how cleanly one T splits the histogram into two groups (Otsu's own score). "
         "Below 0.5 the histogram has no clean valley, so one global T will struggle: try triangle or adaptive."),
        ("🫧", "Blobs", f"{blobs}", "info",
         "Separate white islands. Fewer, bigger blobs means cleaner objects; dozens means salt."),
        ("🧂", "Specks", f"{specks} ({speck_share:.0f} % of blobs)", "check" if speck_share > 50 else "good",
         f"Blobs under min_contour_area = {speck} px², which 4c will throw away. Lower is better. "
         "Specks aren't fatal: 4b's opening sands them off."),
        ("🐘", "Biggest blob", f"{biggest:.0f} % of foreground", "info",
         "High = one dominant object (hopefully the chamber). Low = the foreground is scattered."),
        ("⏱️", "Time", f"{ms:.2f} ms ({1000 / max(ms, 1e-3):.0f} fps)", "info",
         "Lower is better: a clip-on camera has to keep up with the drops. Thresholding is about the cheapest step in the pipeline."),
    ]

    tips = []
    if fg_level == "bad":
        tips.append('Flip `"invert"` first: it\'s the most common reason for an empty or full mask.')
    if knobs["threshold"] in ("global", "otsu") and eta < 0.5:
        tips.append('One big hill and no valley: try `TRY = {"threshold": "triangle"}` or `"adaptive"`.')
    if knobs["threshold"] == "global":
        tips.append(f'Otsu would have chosen T = {otsu_T:.0f}. Try `"global_value": {otsu_T:.0f}` and compare.')
    if adaptive:
        tips.append('Make `"adaptive_block"` bigger (51, 101): fewer rims, and it behaves more like one global T.')
    tips += ['Same knobs, another frame: `FRAME = "low_fluid"`. Does one rule fit every frame?',
             "Try all four methods, then compare the results files side by side.",
             "Next: ▶ `sandbox_4b_morphology.py` repairs this mask."]

    story = (f"👾 I took **{src.name}** as stage 3 passed it on ({gray.shape[1]}×{gray.shape[0]} px) and sorted every "
             f"pixel into two bins with the **{knobs['threshold']}** brain of `recipe.threshold()`. "
             + ("Dark pixels went into the object bin (white), because `invert` is on. " if knobs["invert"]
                else "Bright pixels went into the object bin (white). ")
             + f"**{fg:.1f} %** of the pixels ended up as object.")
    log = sb.write_results(KEY, STEP, "✂️ Stage 4a · Thresholding", story,
                           files=[("📥 input", sb.rel(src)), ("📤 output", sb.rel(out)), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips)
    sb.report(metrics, [out, look], log, knobs, defaults, NAMES)


def separability(gray: np.ndarray, T: float) -> float:
    """Otsu's η = between-group variance / total variance for the split at T. 1.0 = two perfectly separate hills."""
    g = gray.astype(np.float64).ravel()
    lo, hi = g[g <= T], g[g > T]
    if lo.size == 0 or hi.size == 0 or g.var() == 0:
        return 0.0
    w0, w1 = lo.size / g.size, hi.size / g.size
    return float(w0 * w1 * (lo.mean() - hi.mean()) ** 2 / g.var())


if __name__ == "__main__":
    main()
