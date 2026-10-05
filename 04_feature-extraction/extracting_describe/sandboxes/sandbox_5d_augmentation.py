"""
🧬 Stage 5d · Data Augmentation Sandbox
═══════════════════════════════════════
One real frame in, made-up training frames out, with the same recipe.augment() CI runs on every training frame.

    📥 in   your snap, exactly as stage 4 · Segmenting passed it on (build/segmenting/)
    👀 look build/sandbox/stage_5d_look.png        every effect on its own, at its limit · random copies like CI makes
    📝 log  build/sandbox/<unix time>_results_stage_5d.md

Press ▶ in PyCharm (pick "5d · Augmentation" in the run menu), or from the repository root:

    python 04_feature-extraction/extracting_describe/sandboxes/sandbox_5d_augmentation.py
    python 04_feature-extraction/extracting_describe/sandboxes/sandbox_5d_augmentation.py drop

👾 We have 270 training frames and want more, without filming. So the Robot makes plausible *different*
   camera frames of the same situation: mirrored, tilted a few degrees, zoomed in or out a little, a bit
   brighter or darker. Every move is one 2×3 matrix (Euler's affine map: straight lines stay straight).
   The rule: only changes the real camera could see. A drip chamber is never upside down, so flipping
   top ↔ bottom teaches the model something that never happens. And only the TRAINING frames get copies:
   the test set must look like the real world, not like our tricks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "run_pipeline.py").exists())))
from stages import sandbox as sb  # 🔒 the plumbing every sandbox shares (it checks your packages first)

import time

import numpy as np

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#    🎛️  TINKER ZONE — change one thing, press ▶, compare the two results files
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
FRAME = "snap"   # "snap" = your Step 0 photo · "drop", "no_drop", "low_fluid" = a test frame of that class
TRY = {}         # empty = exactly what CI does, with the knobs in ../action.yaml. Then try ONE of these:
#   "augment_rotate_degrees": 45     rotation: tilted up to ±45°. Still a believable clip-on camera?
#   "augment_flip_vertical": True    flipping top ↔ bottom: drops falling upwards
#   "augment_scale": [0.5, 1.5]      scaling: zoomed out to half, in to one and a half
#   "augment_brightness": 0.5        up to 50 % brighter or darker
#   "augment_copies": 3              copies per training frame: 270 frames become 1080
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "extracting", "5d"
NAMES = ["augment_copies", "augment_flip_horizontal", "augment_flip_vertical", "augment_rotate_degrees",
         "augment_scale", "augment_brightness"]   # this sub-step's knobs
COPIES = 6                                         # random copies to show (CI makes augment_copies per training frame)


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)
    r = knobs["augment_rotate_degrees"] or 0
    lo, hi = knobs["augment_scale"] or [1.0, 1.0]
    b = knobs["augment_brightness"] or 0

    # 1. 📥 Load the frame stage 4 · Segmenting passed on, and the training frames to compare it with
    img, space, src = sb.frame("segmenting", FRAME)
    frames = sb.load_images("segmenting")
    train = [i for i, row in enumerate(frames.rows) if row["split"] == "train"]

    # 2. 🎨 Every effect on its own, at the limit of its knob: the recipe's building blocks
    def on(flag: bool) -> str:
        return "" if flag else " (off in CI)"
    effects = [("original", img),
               ("flip <->" + on(knobs["augment_flip_horizontal"]), recipe.flip(img, True, False)),
               ("flip up/down" + on(knobs["augment_flip_vertical"]), recipe.flip(img, False, True)),
               (f"rotate -{r} deg", recipe.affine(img, -r, 1.0)), (f"rotate +{r} deg", recipe.affine(img, r, 1.0)),
               (f"zoom x{lo}", recipe.affine(img, 0, lo)), (f"zoom x{hi}", recipe.affine(img, 0, hi)),
               (f"darker x{1 - b:.2f}", recipe.brightness(img, 1 - b)), (f"brighter x{1 + b:.2f}", recipe.brightness(img, 1 + b))]

    # 3. 🎲 Random copies, the way CI makes them: recipe.augment() rolls the dice for every knob
    rng = np.random.default_rng(knobs.get("seed", 42))
    t0 = time.perf_counter()
    copies = [recipe.augment(img, knobs, rng) for _ in range(COPIES)]
    ms = 1000 * (time.perf_counter() - t0) / COPIES
    panels = [(c, sb.to_luma(e, space)) for c, e in effects] + [(f"random copy {i + 1}", sb.to_luma(c, space))
                                                                for i, c in enumerate(copies)]
    look = sb.look("stage_5d_look.png", panels, cols=5)

    # 4. 📏 Is a copy a believable new frame? Compare how far its features move with how far real frames differ
    def describe(frame):
        return np.concatenate(list(recipe.extract(sb.to_luma(frame, space), knobs).values()))
    me = describe(img)
    shift = float(np.median([np.linalg.norm(describe(c) - me) for c in copies]))
    by_class = {k: [i for i in train if frames.rows[i]["label"] == k] for k in frames.classes}
    dist = {k: float(np.median([np.linalg.norm(describe(frames.images[i]) - me) for i in idx[:40]]))
            for k, idx in by_class.items() if idx}
    row = next((x for x in frames.rows if x["file"] == src.name), None)
    own = row["label"] if row and row["label"] in dist else min(dist, key=dist.get)   # the snap: its closest class
    same = dist[own]
    other = min(v for k, v in dist.items() if k != own) if len(dist) > 1 else float("inf")

    # 5. 🚦 Judge the numbers: a copy should move no further than real frames of its class differ
    ratio = shift / max(same, 1e-9)
    if ratio <= 1.0:
        level, why = "good", f"A copy moves less than two real {own} frames differ ({ratio:.2f}×): a believable new frame."
    elif ratio <= 1.3:
        level, why = "check", f"A copy moves a bit further than real {own} frames differ ({ratio:.2f}×). Plausible, but bold."
    else:
        level, why = "bad", (f"A copy moves {ratio:.1f}× further than real {own} frames differ: the model learns from "
                             "frames that look like nothing the camera will ever see. Tone the knobs down.")
    close = other <= 1.05 * same
    n_train = len(train)
    metrics = [
        ("🧪", "Training frames", f"{n_train} → {n_train * (1 + (knobs['augment_copies'] or 0))}", "info",
         f"augment_copies = {knobs['augment_copies']} extra copies per training frame. The {len(frames.rows) - n_train} "
         "test frames (and your snap) never get copies."),
        ("📐", "How far a copy moves", f"{shift:.2f}", level, why),
        ("📏", f"Two real {own} frames", f"{same:.2f} apart", "info",
         "Median distance between this frame's features and real training frames of its class: natural variation."),
        ("🚧", "A frame of another class", f"{other:.2f} apart", "info",
         "Median distance to the closest other class. "
         + ("About as far as frames of the same class: the class lives in a small detail (the drop), most numbers "
            "describe the chamber. Augmentation must not wipe that detail out." if close else
            "Further than frames of the same class: the classes really are apart in feature space.")),
        ("⏱️", "Time per copy", f"{ms:.2f} ms", "info", "One affine warp: cheap. Copies cost training time later, not here."),
    ]
    if knobs["augment_flip_vertical"]:
        metrics.insert(1, ("🙃", "Vertical flip is on", "drops fall upwards", "bad",
                           "No clip-on camera ever sees that. The model now learns a situation that can't happen."))

    tips = ['Push one knob too far, e.g. `"augment_rotate_degrees": 45`. When does "How far a copy moves" turn 🔴?',
            '`"augment_flip_vertical": True`: look at the drop in the copies. Then check what CI\'s accuracy says.',
            'More is not always better: `"augment_copies": 3` triples the training frames. Does the test accuracy follow? '
            "▶ `recipe.py` tells you in seconds.",
            "Why may only the training frames get copies? One sentence; it's an exam question."]

    story = (f"👾 I took **{src.name}** and made it look like other camera frames with the recipe's building blocks: "
             f"flips, rotations up to ±{r}°, zooms from ×{lo} to ×{hi}, brightness ±{100 * b:.0f} %. Then "
             f"`recipe.augment()` rolled the dice {COPIES} times, the way CI does for every training frame. A copy's "
             f"features move **{shift:.2f}**; two real {own} frames are {same:.2f} apart, a frame of another class "
             f"{other:.2f}.")
    log = sb.write_results(KEY, STEP, "🧬 Stage 5d · Data Augmentation", story,
                           files=[("📥 input", sb.rel(src)), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips)
    sb.report(metrics, [look], log, knobs, defaults, NAMES)


if __name__ == "__main__":
    main()
