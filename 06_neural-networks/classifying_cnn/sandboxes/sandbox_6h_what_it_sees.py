"""
🧠 Stage 6h · What It Learned Sandbox
═════════════════════════════════════
A trained network and one real frame in: the filters it learned itself, what each layer sees in your frame,
and its verdict. Uses 6g's trained network (or trains one with the recipe). Needs TensorFlow.

    📥 in   build/sandbox/stage_6g_model.keras (from 6g) and your snap's pixels (build/extracting/features.npz)
    👀 look build/sandbox/stage_6h_look.png        the frame · 16 learned filters · what layer 1 and 2 see · the verdict
    📝 log  build/sandbox/<unix time>_results_stage_6h.md

Press ▶ in PyCharm (pick "6h · What it learned" in the run menu), or from the repository root:

    python 06_neural-networks/classifying_cnn/sandboxes/sandbox_6h_what_it_sees.py
    python 06_neural-networks/classifying_cnn/sandboxes/sandbox_6h_what_it_sees.py low_fluid

👾 Nobody told the network what an edge is. In 6e we picked Sobel's filter by hand; here every one of the
   first layer's 3×3 filters started as random numbers and was nudged, epoch after epoch, into whatever
   helped tell drop from no_drop from low_fluid. Some end up looking a lot like Sobel (Hubel & Wiesel found
   edge detectors like these in a cat's visual cortex in 1962). The deeper layers combine them into
   patterns nobody has a name for. Bright in a feature map = "found my pattern here".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "run_pipeline.py").exists())))
from stages import sandbox as sb  # 🔒 the plumbing every sandbox shares (it checks your packages first)

import json

import cv2
import numpy as np

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#    🎛️  TINKER ZONE — change one thing, press ▶, compare the two results files
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
FRAME = "snap"   # "snap" = your Step 0 photo · "drop", "no_drop", "low_fluid" = a test frame of that class
TRY = {}         # empty = the network CI trains. Any change here retrains it (6g's saved network no longer fits):
#   "epochs": 10           a network that stopped early: are its filters cleaner, or still noise?
#   "filters": 8           half the filters on the first floor: which ones does it keep?
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "neural_network", "6h"
NAMES = ["epochs", "learning_rate", "batch_size", "dropout", "head", "batch_norm", "conv_layers", "filters", "input_downscale"]
SOBEL = {"Sobel x (vertical edges)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
         "Sobel y (horizontal edges)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)}


def mosaic(maps: np.ndarray, cols: int = 4, size: int = 96, stretch: bool = True) -> np.ndarray:
    """Up to 16 maps (height × width × n) as one grid. stretch: each map to its own brightest pixel (feature maps);
    otherwise the values are shown as they are (0 … 255)."""
    tiles = []
    for k in range(min(16, maps.shape[-1])):
        m = maps[..., k].astype(np.float32)
        if stretch:
            m = 255 * m / m.max() if m.max() > 0 else np.zeros_like(m)
        m = np.uint8(np.clip(m, 0, 255))
        tiles.append(cv2.copyMakeBorder(cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST), 2, 2, 2, 2,
                                        cv2.BORDER_CONSTANT, value=255))
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % cols)
    return np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])


def main() -> None:
    recipe = sb.recipe(KEY)
    tf = recipe.tf
    knobs, defaults = sb.knobs(KEY, TRY)

    # 1. 📥 A trained network: 6g's, if it was trained with these knobs on these frames; else train one now
    data = sb.features()
    saved, made = sb.OUT / "stage_6g_model.keras", sb.OUT / "stage_6g_model.json"
    npz = sb.ROOT / "build" / "extracting" / "features.npz"
    same = (saved.exists() and made.exists() and saved.stat().st_mtime >= npz.stat().st_mtime
            and json.loads(made.read_text()) == json.loads(json.dumps({k: knobs[k] for k in NAMES}, default=str)))
    if same:
        model, source = tf.keras.models.load_model(saved), "6g's trained network"
    else:
        print("👾 6g's network doesn't match these knobs or frames (or isn't there yet), so I'm training one: 10–60 s…")
        I_train = data.I_train[..., None] if data.I_train.ndim == 3 else data.I_train
        model = recipe.build_network(I_train.shape[1:], len(data.classes), knobs, data.seed)
        recipe.train(model, I_train, data.y_train, knobs, data.seed, verbose=0)
        source = "a network trained just now"

    # 2. 🔦 Walk your frame through the network, layer by layer, and keep what every ReLU let through
    _, frame, true, caption = sb.pick(data, FRAME)
    x = (frame[..., None] if frame.ndim == 2 else frame)[None].astype(np.float32)
    seen = []
    for layer in model.layers:
        x = layer(x, training=False)
        if type(layer).__name__ == "Activation":
            seen.append(np.asarray(x)[0])
    proba = np.asarray(x)[0]
    verdict = int(np.argmax(proba))

    # 3. 🧩 The first layer's filters: 3×3 weights it learned, compared with Sobel's (6e)
    kernels = next(l for l in model.layers if type(l).__name__ == "Conv2D").get_weights()[0][:, :, 0, :]   # 3 × 3 × filters
    likeness = []
    for k in range(kernels.shape[-1]):
        w = kernels[..., k].ravel()
        scores = {n: float(abs(w @ s.ravel()) / max(np.linalg.norm(w) * np.linalg.norm(s), 1e-9)) for n, s in SOBEL.items()}
        likeness.append(max(scores.items(), key=lambda kv: kv[1]))
    sobel_like = sum(score >= 0.8 for _, score in likeness)
    quiet = [k for k in range(seen[0].shape[-1]) if seen[0][..., k].max() <= 0]

    # 4. 👀 The frame, the filters, what layers 1 and 2 see, and the verdict
    m = max(float(np.abs(kernels).max()), 1e-9)
    filt = np.uint8(np.clip(127.5 + 127.5 * kernels / m, 0, 255))                # grey = 0, light = +, dark = −
    fig = sb.new_figure(4.2, 3.6)
    ax = fig.add_subplot()
    ax.barh(data.classes, proba, color=["#2e7d32" if i == verdict else "#bdbdbd" for i in range(len(proba))])
    ax.set_xlim(0, 1.15)
    for i, p in enumerate(proba):
        ax.text(p, i, f" {p:.0%}", va="center", fontsize=10)
    ax.set_title("its verdict", fontsize=10)
    fig.tight_layout()
    look = sb.look("stage_6h_look.png", [(caption, sb.bigger(frame, data.space)),
                                         ("16 learned filters (grey = 0)", mosaic(filt, size=72, stretch=False)),
                                         ("what layer 1 sees", mosaic(seen[0])),
                                         ("what layer 2 sees", mosaic(seen[1]) if len(seen) > 1 else mosaic(seen[0])),
                                         (f"verdict: {data.classes[verdict]}", sb.figure(fig))])

    # 5. 🚦 Judge the numbers
    right = None if true is None else verdict == true
    metrics = [
        ("🏛️", "Verdict", data.classes[verdict] + ("" if right is None else (" ✅" if right else f" ❌ (truth: {data.classes[true]})")),
         "info" if right is None else "good" if right else "bad", f"What {source} says about {caption}."),
        ("🎯", "How sure", f"{proba[verdict]:.0%}", "good" if proba[verdict] >= 0.7 else "check",
         "Softmax probability. A network can be very sure and still wrong: sureness is not correctness."),
        ("🧭", "Filters that look like Sobel", f"{sobel_like} of {kernels.shape[-1]}", "info",
         "Learned filters whose 3×3 weights point the same way as Sobel x or y (cosine similarity ≥ 0.8, either sign). "
         "Nobody told it about edges."),
        ("😴", "Quiet filters on this frame", f"{len(quiet)} of {seen[0].shape[-1]}", "check" if len(quiet) > seen[0].shape[-1] // 2 else "info",
         "Filters that didn't fire anywhere in this frame: their pattern isn't here (or they never learned one)."),
    ]
    table = ["## 🧩 The first layer's filters", "", "| Filter | Looks most like | Similarity |", "|---|---|---|"]
    table += [f"| {k + 1} | {name} | {score:.2f} |" for k, (name, score) in enumerate(likeness)]
    tips = ['Compare with 6e: which learned filter is closest to Sobel y, and does it light up on the fluid line in "what layer 1 sees"?',
            'Another frame: `FRAME = "low_fluid"`. Which feature maps change most between classes?',
            "Is it sure and wrong on your snap? The test frames were synthetic; your snap is a real photo. "
            "That's domain shift: ❓ How to train our model on cows?",
            "Ship it? ▶ `recipe.py` runs the network the way CI does, exam and gates included."]

    story = (f"👾 I opened {source} and walked **{caption}** through it, layer by layer. Of its first layer's "
             f"{kernels.shape[-1]} filters, **{sobel_like}** learned something close to Sobel's edge filters on their own; "
             f"{len(quiet)} stayed quiet on this frame. Its verdict: **{data.classes[verdict]}**, {proba[verdict]:.0%} sure"
             + ("." if right is None else (". Right. ✅" if right else f". Wrong: it's {data.classes[true]}. ❌")))
    log = sb.write_results(KEY, STEP, "🧠 Stage 6h · What It Learned", story,
                           files=[("📥 input", sb.rel(saved) if same else "build/extracting/features.npz"), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips, extra="\n".join(table))
    sb.report(metrics, [look], log, knobs, defaults, NAMES)


if __name__ == "__main__":
    main()
