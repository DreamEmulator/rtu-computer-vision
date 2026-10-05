"""
🧠 Stage 6f · The Network Sandbox
═════════════════════════════════
The network CI trains, built by the same recipe.build_network(), not trained yet: its layers, its shapes,
and where its parameters are. Needs TensorFlow: pip install -r requirements-cnn.txt

    📥 in   the shape of stage 4's pixels and the classes (build/extracting/features.npz)
    👀 look build/sandbox/stage_6f_look.png        numbers per layer · parameters per layer
    📝 log  build/sandbox/<unix time>_results_stage_6f.md   with the layer table

Press ▶ in PyCharm (pick "6f · The network" in the run menu), or from the repository root:

    python 06_neural-networks/classifying_cnn/sandboxes/sandbox_6f_network.py

👾 Floors of a building. On the ground floor, filters like 6e's look at 3×3 pixels. Pooling halves the
   picture, so on the next floor the same 3×3 filters cover twice as much of the frame, and there are
   twice as many of them. By the top floor every neuron sees a big part of the frame, in only a few
   pixels. Then the head turns that into one probability per class. A parameter is one weight the
   network will learn; Rosenblatt's 1958 perceptron had 400 of them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "run_pipeline.py").exists())))
from stages import sandbox as sb  # 🔒 the plumbing every sandbox shares (it checks your packages first)

import numpy as np

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#    🎛️  TINKER ZONE — change one thing, press ▶, compare the two results files
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
TRY = {}         # empty = exactly the network CI trains, with the knobs in ../action.yaml. Then try ONE of these:
#   "head": "gap"            global average pooling instead of flatten: watch the parameter count
#   "conv_layers": 4         one floor more: smaller pictures at the top, a wider view per neuron
#   "filters": 32            twice the filters on every floor
#   "input_downscale": 1     start from 128×128 instead of 64×64
#   "dense_units": 16        a smaller head
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "neural_network", "6f"
NAMES = ["input_downscale", "conv_layers", "filters", "batch_norm", "head", "dense_units", "dropout"]


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)

    # 1. 📥 What goes in: stage 4's pixels, one channel; what comes out: one probability per class
    data = sb.features()
    shape = data.I_train.shape[1:] + ((1,) if data.I_train.ndim == 3 else ())

    # 2. 🏗️ Build it with the recipe (no training: every weight is still random)
    model, ms = sb.timed(recipe.build_network, shape, len(data.classes), knobs, data.seed)

    # 3. 📋 Floor by floor: its output shape, how many numbers that is, how many weights it will learn
    rows, field, jump = [], 1, 1
    for layer in model.layers:
        kind = type(layer).__name__
        out = tuple(int(v) for v in layer.output.shape[1:])
        if kind == "Conv2D":                                          # how far one neuron sees, in input pixels
            field += 2 * jump
        elif kind in ("MaxPooling2D", "AveragePooling2D"):
            field += (layer.pool_size[0] - 1) * jump
            jump *= layer.pool_size[0]
        rows.append({"layer": kind, "shape": "×".join(map(str, out)), "numbers": int(np.prod(out)),
                     "params": int(layer.count_params()), "field": field if len(out) == 3 else None})
    total = int(model.count_params())
    convs = [r for r in rows if r["layer"] == "Conv2D"]
    dense = [r for r in rows if r["layer"] == "Dense"]
    head_share = 100 * dense[0]["params"] / max(total, 1) if dense else 0.0
    top = [r for r in rows if r["field"]][-1]

    # 4. 👀 Numbers per floor (the picture shrinks) and parameters per floor (where the learning sits)
    fig = sb.new_figure(12, 4.2)
    a, b = fig.subplots(1, 2)
    labels = [f"{i} {r['layer']}" for i, r in enumerate(rows)]
    a.barh(labels, [r["numbers"] for r in rows], color="#5c6bc0")
    a.set_xscale("log"); a.invert_yaxis(); a.set_title("numbers coming out of each layer (log)", fontsize=10)
    b.barh(labels, [r["params"] for r in rows], color=["#e65100" if r["layer"] == "Dense" else "#8bc34a" for r in rows])
    b.invert_yaxis(); b.set_title(f"weights to learn: {total:,} (orange = the head)", fontsize=10)
    for ax in (a, b):
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    look = sb.look("stage_6f_look.png", [("the network, floor by floor", sb.figure(fig))])

    # 5. 🚦 Judge the numbers
    n_train = len(data.y_train)
    metrics = [
        ("⚖️", "Weights to learn", f"{total:,}", "check" if total > 100 * n_train else "info",
         f"From {n_train} training frames (augmented copies included): about {total / max(n_train, 1):,.0f} weights per frame. "
         "Far more weights than examples is an invitation to memorise. 6g will show whether it accepts."),
        ("🧠", "In the head", f"{head_share:.0f} % of all weights", "check" if head_share > 50 else "good",
         "The first Dense layer connects every number of the top floor to every neuron of the head. With flatten that's "
         "most of the network; with `head: gap` it shrinks to a handful."),
        ("👁️", "What a top-floor neuron sees", f"{top['field']}×{top['field']} px of the frame", "info",
         f"Its receptive field, on the {shape[0]}×{shape[1]} frame. Each conv adds 2 steps, each pooling doubles the step."),
        ("🏢", "Floors", f"{len(convs)} conv layers: " + " → ".join(r["shape"] for r in convs), "info",
         "Height × width × filters after each convolution: the picture shrinks, the number of filters grows."),
        ("📦", "Size", f"{4 * total / 1024:,.0f} KB", "info", "4 bytes per weight (32-bit floats). Quantized tflite: about a quarter."),
        ("⏱️", "Time to build", f"{ms:.0f} ms", "info", "Building is instant. Training (6g) is where the time goes."),
    ]
    table = ["## 🏢 Floor by floor", "", "| # | Layer | Output | Numbers | Weights | Sees (px) |", "|---|---|---|---|---|---|"]
    table += [f"| {i} | {r['layer']} | {r['shape']} | {r['numbers']:,} | {r['params']:,} | {r['field'] or ''} |"
              for i, r in enumerate(rows)]
    tips = []
    if head_share > 50:
        tips.append('Most weights sit in the head: try `"head": "gap"`. How many weights are left? Does it still learn (6g)?')
    tips += ['`"conv_layers": 4`: what happens to "What a top-floor neuron sees"?',
             f"The forest has no weights to learn like this. Which would you trust with {n_train} frames, and why?",
             "Next: ▶ `sandbox_6g_training.py` trains this network and draws its learning curve."]

    story = (f"👾 I built the network CI trains with `recipe.build_network()`: {len(convs)} floors of filters on a "
             f"{shape[0]}×{shape[1]} frame, then a `{knobs['head']}` head and {len(data.classes)} outputs. It has "
             f"**{total:,} weights** to learn, {head_share:.0f} % of them in the head. A neuron on the top floor sees "
             f"{top['field']}×{top['field']} px of the frame.")
    log = sb.write_results(KEY, STEP, "🧠 Stage 6f · The Network", story,
                           files=[("📥 input", "build/extracting/features.npz"), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips, extra="\n".join(table))
    sb.report(metrics, [look], log, knobs, defaults, NAMES)


if __name__ == "__main__":
    main()
