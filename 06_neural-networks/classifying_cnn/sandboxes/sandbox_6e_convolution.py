"""
🧠 Stage 6e · One Filter by Hand Sandbox
════════════════════════════════════════
One real frame in, one feature map out: what ONE filter of the network's first layer does, by hand.
No TensorFlow needed: this is plain OpenCV and NumPy.

    📥 in   your snap, exactly as the network receives it (stage 4's pixels, from build/extracting/)
    👀 look build/sandbox/stage_6e_look.png        the frame → the filter → ReLU → max pooling
    📝 log  build/sandbox/<unix time>_results_stage_6e.md

Press ▶ in PyCharm (pick "6e · One filter" in the run menu), or from the repository root:

    python 06_neural-networks/classifying_cnn/sandboxes/sandbox_6e_convolution.py
    python 06_neural-networks/classifying_cnn/sandboxes/sandbox_6e_convolution.py low_fluid

👾 A convolution slides a small grid of weights (the filter, or kernel) over the frame. At every spot it
   multiplies the 3×3 pixels under it by the 9 weights and adds them up: one number, large where the
   frame looks like the filter. That's a feature map. ReLU then keeps the positive answers and zeroes
   the rest ("did it fire?"), and max pooling keeps the strongest answer in every 2×2 patch: half the
   width, half the height, the same story. HOG (5a) used a filter like this too, Sobel's, designed by a
   person in 1968. The network's first layer has 16 of them, and it learns their weights itself (6h).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "run_pipeline.py").exists())))
from stages import sandbox as sb  # 🔒 the plumbing every sandbox shares (it checks your packages first)

import time

import cv2
import numpy as np

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#    🎛️  TINKER ZONE — change one thing, press ▶, compare the two results files
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
FRAME = "snap"         # "snap" = your Step 0 photo · "drop", "no_drop", "low_fluid" = a test frame of that class
FILTER = "sobel_y"     # "sobel_x"   brightness changes left ↔ right: vertical edges (the chamber's walls)
#                        "sobel_y"   brightness changes top ↔ bottom: horizontal edges (the fluid line, a drop)
#                        "laplace"   changes in every direction: spots and outlines
#                        "blur"      the average of 9 pixels: smooths, finds nothing
#                        or your own 3×3 weights, e.g. [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
TRY = {}               # empty = the network's knobs in ../action.yaml. Then try:
#   "input_downscale": 1   the network's first step averages 2×2 pixels into one (128 → 64). 1 = keep every pixel
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "neural_network", "6e"
NAMES = ["input_downscale"]
FILTERS = {"sobel_x": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], "sobel_y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
           "laplace": [[0, 1, 0], [1, -4, 1], [0, 1, 0]], "blur": [[1 / 9] * 3] * 3}


def pool(img: np.ndarray, k: int, how) -> np.ndarray:
    """k×k pooling: cut the frame into k×k patches, keep one number per patch (np.mean or np.max)."""
    h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k
    return how(img[:h, :w].reshape(h // k, k, w // k, k), axis=(1, 3))


def show(fmap: np.ndarray) -> np.ndarray:
    """A feature map as a picture: 0 is mid-grey, positive is light, negative is dark."""
    m = max(float(np.abs(fmap).max()), 1e-9)
    return np.uint8(np.clip(127.5 + 127.5 * fmap / m, 0, 255))


def main() -> None:
    knobs, defaults = sb.knobs(KEY, TRY)
    kernel = np.array(FILTERS.get(FILTER, FILTER) if isinstance(FILTER, str) else FILTER, dtype=np.float32)
    if kernel.shape != (3, 3):
        raise SystemExit(f'👾 FILTER must be one of {", ".join(FILTERS)} or a 3×3 list of numbers, not {FILTER!r}.')

    # 1. 📥 The frame exactly as the network receives it: stage 4's pixels, scaled to 0 … 1 like its first layer does
    data = sb.features()
    _, frame, true, caption = sb.pick(data, FRAME)
    x = sb.to_luma(frame, data.space).astype(np.float32) / 255.0
    k = int(knobs["input_downscale"] or 1)
    if k > 1:
        x = pool(x, k, np.mean)                                   # the network's AveragePooling2D

    # 2. 🧮 One filter, slid over every pixel: 9 multiplications and a sum per pixel
    t0 = time.perf_counter()
    fmap = cv2.filter2D(x, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)   # "same" padding with zeros, like Conv2D
    fired = np.maximum(fmap, 0)                                   # ReLU
    pooled = pool(fired, 2, np.max)                               # MaxPooling2D
    ms = 1000 * (time.perf_counter() - t0)

    # 3. 👀 Every step, enlarged to the same size so you can compare them
    def big(img):
        return cv2.resize(img, (384, 384), interpolation=cv2.INTER_NEAREST)
    yy, xx = np.unravel_index(np.argmax(fired), fired.shape)
    fire_vis = cv2.cvtColor(big(np.uint8(255 * fired / max(fired.max(), 1e-9))), cv2.COLOR_GRAY2BGR)
    s = 384 / fired.shape[0]
    cv2.circle(fire_vis, (int((xx + 0.5) * s), int((yy + 0.5) * s)), 14, (31, 140, 240), 2)
    kernel_vis = big(show(kernel))
    for i in range(3):
        for j in range(3):
            cv2.putText(kernel_vis, f"{kernel[i, j]:+.2g}", (j * 128 + 30, i * 128 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0) if kernel[i, j] >= 0 else (255, 255, 255), 2, cv2.LINE_AA)
    look = sb.look("stage_6e_look.png", [
        (f"input {x.shape[1]}x{x.shape[0]}", big(np.uint8(255 * x))), (f"the filter: {FILTER}" if isinstance(FILTER, str) else "your filter", kernel_vis),
        ("feature map (grey = 0)", big(show(fmap))), ("ReLU: what fired", fire_vis),
        (f"max pooling {pooled.shape[1]}x{pooled.shape[0]}", big(np.uint8(255 * pooled / max(pooled.max(), 1e-9))))])

    # 4. 🚦 Judge the numbers
    zeroed = 100 * float((fmap <= 0).mean())
    metrics = [
        ("🔢", "Multiplications", f"{9 * fmap.size:,}", "info",
         f"9 per pixel for this one filter on {x.shape[1]}×{x.shape[0]} pixels. The network's first layer has "
         f"{knobs['filters']} filters: {9 * fmap.size * knobs['filters']:,} multiplications for one frame. GPUs exist for this."),
        ("🔥", "Fired", f"{100 - zeroed:.0f} % of the pixels", "info",
         "Where the frame looks like the filter (a positive answer). ReLU zeroes the rest: "
         f"{zeroed:.0f} % of the map carries no signal to the next layer."),
        ("🎯", "Strongest answer", f"at ({xx}, {yy})", "info",
         "Circled in orange. Is it on the drip chamber? Then this filter would be a useful one to learn."),
        ("🗜️", "After pooling", f"{fmap.shape[1]}×{fmap.shape[0]} → {pooled.shape[1]}×{pooled.shape[0]}", "info",
         "A quarter of the numbers, the strongest answers kept. Three conv + pool layers shrink 64 px to 8 px: "
         "the deeper layers see bigger patterns in fewer pixels."),
        ("⏱️", "Time", f"{ms:.2f} ms", "info", "One filter, one frame, on the CPU."),
    ]
    tips = ['`FILTER = "sobel_x"`, then `"sobel_y"`: which one fires on the fluid line, which on the walls?',
            "Make one up: `FILTER = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]`. What does it look for?",
            '`TRY = {"input_downscale": 1}`: every pixel, four times the work. Does the drop get clearer?',
            "Next: ▶ `sandbox_6f_network.py` stacks layers of these filters into the network CI trains."]

    story = (f"👾 I slid the **{FILTER if isinstance(FILTER, str) else 'custom'}** filter over **{caption}** "
             f"({x.shape[1]}×{x.shape[0]} after `input_downscale` {k}): 9 multiplications and a sum at every pixel. "
             f"{100 - zeroed:.0f} % of the pixels fired, the strongest at ({xx}, {yy}). Max pooling kept the strongest "
             f"answer per 2×2 patch: {pooled.shape[1]}×{pooled.shape[0]}. The network does this with {knobs['filters']} "
             "filters at once in its first layer, and learns their weights instead of being told.")
    log = sb.write_results(KEY, STEP, "🧠 Stage 6e · One Filter by Hand", story,
                           files=[("📥 input", "build/extracting/features.npz"), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips)
    sb.report(metrics, [look], log, knobs, defaults, NAMES)


if __name__ == "__main__":
    main()
