"""
🧠 Stage 6g · Training Sandbox
══════════════════════════════
The real pixels in, a trained network out, with the same recipe.build_network() and recipe.train() CI runs.
Needs TensorFlow: pip install -r requirements-cnn.txt

    📥 in   stage 4's pixels for every training and test frame (build/extracting/features.npz)
    👀 look build/sandbox/stage_6g_look.png        accuracy per epoch · loss per epoch
    💾 keep build/sandbox/stage_6g_model.keras     the trained network, for 6h
    📝 log  build/sandbox/<unix time>_results_stage_6g.md

Press ▶ in PyCharm (pick "6g · Training" in the run menu), or from the repository root:

    python 06_neural-networks/classifying_cnn/sandboxes/sandbox_6g_training.py

👾 Learning by being wrong. The network guesses every training frame, measures how wrong it was (the loss),
   and nudges every weight a little in the direction that would have made it less wrong: downhill, like
   Cauchy's gradient descent (1847), with backpropagation telling each weight which way is down (1986).
   One pass over all training frames is an epoch. 15 % of the training frames sit out every epoch: if the
   network gets better on the frames it learns from but not on those, it's memorising, not learning.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "run_pipeline.py").exists())))
from stages import sandbox as sb  # 🔒 the plumbing every sandbox shares (it checks your packages first)

import json
import time

import numpy as np

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#    🎛️  TINKER ZONE — change one thing, press ▶, compare the two results files
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
TRY = {}         # empty = exactly the training CI runs, with the knobs in ../action.yaml. Then try ONE of these:
#   "epochs": 10             fewer passes: does it stop before it starts memorising?
#   "learning_rate": 0.01    bigger steps downhill: faster, or does it overshoot?
#   "learning_rate": 0.0001  tiny steps: does it get anywhere in 40 epochs?
#   "dropout": 0.5           silence half the head's neurons at random while training: harder to memorise
#   "head": "gap"            90 % fewer weights (see 6f): less to memorise with
#   "batch_norm": True       keep every layer's numbers in a trainable range (Ioffe & Szegedy, 2015)
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "neural_network", "6g"
NAMES = ["epochs", "learning_rate", "batch_size", "dropout", "head", "batch_norm", "conv_layers", "filters", "input_downscale"]


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)

    # 1. 📥 Stage 4's pixels, one channel
    data = sb.features()
    I_train, I_test = (a[..., None] if a.ndim == 3 else a for a in (data.I_train, data.I_test))

    # 2. 🏋️ Build and train, with the recipe
    print(f"👾 Training {knobs['epochs']} epochs on {len(data.y_train)} frames. On a laptop that's 10–60 seconds…")
    t0 = time.perf_counter()
    model = recipe.build_network(I_train.shape[1:], len(data.classes), knobs, data.seed)
    history = recipe.train(model, I_train, data.y_train, knobs, data.seed, verbose=0)
    seconds = time.perf_counter() - t0
    y_pred = model.predict(I_test, verbose=0).argmax(axis=1)
    test_acc = float((y_pred == data.y_test).mean())
    model.save(sb.OUT / "stage_6g_model.keras")
    (sb.OUT / "stage_6g_model.json").write_text(json.dumps({k: knobs[k] for k in NAMES}, default=str))

    # 3. 👀 The learning curve: the frames it learns from against the frames that sit out
    acc, val, loss, vloss = (np.array(history[k]) for k in ("accuracy", "val_accuracy", "loss", "val_loss"))
    best = int(np.argmax(val))
    turn = int(np.argmin(vloss))                                     # where it started memorising
    epochs = np.arange(1, len(acc) + 1)
    fig = sb.new_figure(11, 3.9)
    a, b = fig.subplots(1, 2)
    a.plot(epochs, acc, label="frames it learns from", color="#1565c0")
    a.plot(epochs, val, label="frames that sit out", color="#e65100")
    a.axvline(best + 1, color="#999", ls=":", label=f"best sit-out epoch ({best + 1})")
    a.axhline(test_acc, color="#2e7d32", ls="--", label=f"test accuracy {test_acc:.2f}")
    a.set_ylim(0, 1.02); a.set_xlabel("epoch"); a.set_ylabel("accuracy"); a.legend(fontsize=8)
    a.set_title("accuracy", fontsize=10)
    b.plot(epochs, loss, color="#1565c0"); b.plot(epochs, vloss, color="#e65100")
    b.axvline(turn + 1, color="#999", ls=":")
    b.annotate(f"lowest at epoch {turn + 1}", (turn + 1, vloss[turn]), xytext=(8, 30), textcoords="offset points",
               fontsize=8, arrowprops={"arrowstyle": "->", "color": "#999"})
    b.set_xlabel("epoch"); b.set_ylabel("loss (how wrong)"); b.set_title("loss: should go down, together", fontsize=10)
    fig.tight_layout()
    look = sb.look("stage_6g_look.png", [("the learning curve", sb.figure(fig))])

    # 4. 🚦 Judge the numbers
    gap = acc[-1] - val[-1]
    gate = knobs["gate_min_accuracy"]
    forest = sb.ROOT / "build" / "random_forest" / "metrics.json"
    forest_acc = json.loads(forest.read_text())["metrics"].get("test accuracy") if forest.exists() else None
    metrics = [
        ("🎓", "Test accuracy", f"{test_acc:.2f}", "good" if gate is None or test_acc >= gate else "bad",
         f"On the locked test set. 🚦 CI ships the network from {gate}"
         + (f"; the forest scores {forest_acc:.2f}. A network that doesn't beat it doesn't ship." if forest_acc else ".")),
        ("📚", "Accuracy on frames it learns from", f"{acc[-1]:.2f}", "info", "After the last epoch."),
        ("🪑", "Accuracy on frames that sit out", f"{val[-1]:.2f} (best {val[best]:.2f} at epoch {best + 1})", "info",
         "15 % of the training frames it never learns from: a practice test during training."),
        ("📉", "Sit-out loss lowest at", f"epoch {turn + 1} of {len(acc)}", "check" if turn + 1 < 0.75 * len(acc) else "good",
         "After this epoch the network got more sure of its wrong answers on frames it doesn't learn from, while it kept "
         "improving on the ones it does: the moment it started memorising."),
        ("🕳️", "Memorising gap", f"{gap:+.2f}", "bad" if gap > 0.15 else "check" if gap > 0.05 else "good",
         "Learns-from minus sits-out. Big = it memorises the training frames instead of learning what a drop looks like: "
         "overfitting. Fight it with dropout, fewer weights (head: gap), more data (augment_copies in stage 5)."),
        ("⏱️", "Training time", f"{seconds:.0f} s", "info",
         "The forest trains in about a second. AlexNet (2012) needed two gaming GPUs and a week: this is why."),
    ]
    tips = []
    if gap > 0.15:
        tips.append('It memorises: try `"dropout": 0.5`, then `"head": "gap"`. Which one closes the gap?')
    if turn + 1 < 0.75 * len(acc):
        tips.append(f'The sit-out loss was lowest at epoch {turn + 1}: try `"epochs": {turn + 1}`. Stopping there is called '
                    "early stopping. Does the test accuracy agree?")
    tips += ['`"learning_rate": 0.01` and `0.0001`: compare the shapes of the curves. Too big jumps around, too small crawls.',
             "Next: ▶ `sandbox_6h_what_it_sees.py` opens this trained network and shows what its filters learned."]

    story = (f"👾 I built the network with `recipe.build_network()` and trained it with `recipe.train()`: "
             f"{knobs['epochs']} epochs, learning rate {knobs['learning_rate']}, in {seconds:.0f} s. On the frames it "
             f"learns from it ends at **{acc[-1]:.2f}**, on the frames that sit out at **{val[-1]:.2f}**, and on the "
             f"locked test set at **{test_acc:.2f}**" + (f" (the forest: {forest_acc:.2f})." if forest_acc else "."))
    log = sb.write_results(KEY, STEP, "🧠 Stage 6g · Training", story,
                           files=[("📥 input", "build/extracting/features.npz"), ("👀 look", sb.rel(look)),
                                  ("💾 model", sb.rel(sb.OUT / "stage_6g_model.keras"))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips)
    sb.report(metrics, [look], log, knobs, defaults, NAMES)


if __name__ == "__main__":
    main()
