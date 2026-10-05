"""
🧠 Stage 6 · Classifying · Neural Network — can a network learn its own features, straight from the pixels?
══════════════════════════════════════════════════════════════════════════════════════════════════════════
Lecture: Image Classification with Neural Networks · Thu 08.10 · Why: README.md · The values: action.yaml
Needs TensorFlow:  pip install -r requirements-cnn.txt   (and enabled: "true" in action.yaml)

This is the file CI runs for this stage. Press ▶ in PyCharm (or `python 6_neural-network/neural_network.py`):

    build/neural_network/report.md          the values you passed in, every number explained, the gates
    build/neural_network/before_after.png   the confusion matrix · the learning curve · the frames it got wrong

👾 One filter. A convolution slides a 3 × 3 grid of weights over the frame; at every pixel it multiplies the 9
   pixels under it by the 9 weights and adds them up: large where the frame looks like the filter. HOG (stage 5)
   used one like that, Sobel's, designed by a person in 1968. ReLU keeps the positive answers ("did it fire?"),
   max pooling keeps the strongest answer per 2 × 2 patch: half the width and height, the same story.

👾 Floors of a building. The first floor has `filters` of those filters; every next floor has twice as many,
   on a picture half the size, so each neuron on the top floor sees a big part of the frame (44 × 44 px with the
   default values). Then the head turns the top floor into one probability per class. Nobody tells the network
   what the filters should be: they start as random numbers (LeCun et al., 1989).

👾 Learning by being wrong. Guess every training frame, measure how wrong (the loss), nudge every weight a little
   downhill (Cauchy's gradient descent, 1847; backpropagation, 1986; Adam, 2014), repeat for `epochs` passes.
   15 % of the training frames sit out: when the network keeps improving on the frames it learns from but not on
   those, it's memorising, not learning. With ~285 000 weights and 540 frames, it will try. dropout, a smaller
   head (gap) and more data (augment_copies in stage 5) fight back. On a laptop this takes seconds; AlexNet (2012)
   needed two gaming GPUs and a week, and that's why deep learning took off when it did.

   The network is yours to change (build_network, train); the exam isn't (🔒 stages/exam.py).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # 🔒 finds stages/

import json
import time

import numpy as np

from stages import exam                         # 🔒 the exam: the same for every model
from stages.registry import BY_KEY
from stages.stage import Stage, run             # 🔒 the plumbing: values, report, gates

KEY = "neural_network"


def build_network(input_shape: tuple, n_classes: int, knobs: dict, seed: int):
    """Convolution → ReLU → max pooling, `conv_layers` times, then a head that votes."""
    import tensorflow as tf
    layers = tf.keras.layers
    tf.keras.utils.set_random_seed(seed)                       # same seed, same starting weights
    model = tf.keras.Sequential([layers.Input(shape=input_shape), layers.Rescaling(1 / 255.0)])
    if (knobs.get("input_downscale") or 1) > 1:
        model.add(layers.AveragePooling2D(knobs["input_downscale"]))   # fewer pixels: faster, coarser
    for i in range(knobs.get("conv_layers", 3)):
        model.add(layers.Conv2D(knobs.get("filters", 16) * 2 ** i, 3, padding="same", use_bias=not knobs.get("batch_norm")))
        if knobs.get("batch_norm"):
            model.add(layers.BatchNormalization())             # keep every layer's numbers in a trainable range (2015)
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D())
    model.add(layers.GlobalAveragePooling2D() if knobs.get("head", "flatten") == "gap" else layers.Flatten())
    model.add(layers.Dense(knobs.get("dense_units", 64), activation="relu"))
    model.add(layers.Dropout(knobs.get("dropout", 0.3)))       # silence neurons at random while training: harder to memorise
    model.add(layers.Dense(n_classes, activation="softmax"))   # one probability per class
    return model


def train(model, I_train: np.ndarray, y_train: np.ndarray, knobs: dict, seed: int, verbose: int = 2) -> dict:
    """Gradient descent with Adam; 15 % of the training frames sit out to watch for memorising. → the history."""
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(knobs.get("learning_rate", 1e-3)),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    order = np.random.default_rng(seed).permutation(len(y_train))      # mix the classes before the sit-out split
    history = model.fit(I_train[order], y_train[order], validation_split=0.15, epochs=knobs.get("epochs", 40),
                        batch_size=knobs.get("batch_size", 32), verbose=verbose)
    return history.history


def main() -> None:
    stage = Stage(KEY)                           # the values from action.yaml
    c = stage.knobs
    if not c.get("enabled"):
        stage.report.status_override = "off"
        stage.note("Switched off: set enabled to \"true\" in 6_neural-network/action.yaml (and pip install -r requirements-cnn.txt)")
        stage.finish("6 · Neural Network — off")
        return
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        raise SystemExit("❌ The neural network needs TensorFlow → pip install -r requirements-cnn.txt")

    seed = stage.cfg["digital_data"].get("seed", 42)
    d = exam.load_features()                     # stage 5's output; the network takes the pixels, I_*
    for k in ("I_train", "I_test", "I_snap"):
        if getattr(d, k).ndim == 3:
            setattr(d, k, getattr(d, k)[..., None])  # one channel

    t0 = time.time()
    model = build_network(d.I_train.shape[1:], len(d.classes), c, seed)
    history = train(model, d.I_train, d.y_train, c, seed)
    train_s = time.time() - t0
    t1 = time.perf_counter()
    y_pred = model.predict(d.I_test, verbose=0).argmax(axis=1)
    predict_ms = 1000 * (time.perf_counter() - t1) / len(d.y_test)
    snap_proba = model.predict(d.I_snap, verbose=0)[0] if len(d.I_snap) else None
    model.save(stage.dir / "model.keras")

    acc, val, vloss = (np.array(history[k]) for k in ("accuracy", "val_accuracy", "val_loss"))
    turn = int(np.argmin(vloss))
    head = sum(l.count_params() for l in model.layers if type(l).__name__ == "Dense")
    stage.metric("final validation accuracy", history["val_accuracy"][-1],
                 "On the 15 % of training frames that sat out: a practice test during training.")
    stage.metric("parameters", int(model.count_params()),
                 f"Weights it learned. {100 * head / model.count_params():.0f} % of them sit in the head (Dense layers): "
                 "head: gap cuts most of them.", "check" if model.count_params() > 100 * len(d.y_train) else "info")
    stage.metric("memorising gap", float(acc[-1] - val[-1]),
                 "Accuracy on the frames it learns from minus on the frames that sit out. Big = memorising (overfitting).",
                 "bad" if acc[-1] - val[-1] > 0.15 else "check" if acc[-1] - val[-1] > 0.05 else "good")
    stage.metric("sit-out loss lowest at epoch", turn + 1,
                 f"After this epoch (of {len(acc)}) it got more sure of its wrong answers on frames it doesn't learn from: "
                 "where it started memorising. epochs: that number is early stopping.",
                 "check" if turn + 1 < 0.75 * len(acc) else "good")
    if (export := c.get("export") or "none") != "none":
        lite_pred = export_tflite(model, stage.dir, export, d.I_test, d.classes, stage.cfg)
        stage.metric("tflite test accuracy", float(np.mean(lite_pred == d.y_test)), "The exported FILE, graded on the test set, the way the phone will run it.")
        stage.metric("tflite agrees with keras %", 100 * float(np.mean(lite_pred == y_pred)), "Should be 100, or very close for tflite_quantized.")
        stage.perf("tflite size KB", (stage.dir / "model.tflite").stat().st_size / 1024)
        stage.note("model.tflite and preprocessing.json are in the neural_network artifact")
    stage.metric("model", "cnn")

    # 🎓 The exam (🔒 the same for every model)
    exam.take(stage, d, y_pred, snap_proba, stage.dir / "model.keras", train_s, predict_ms,
              lambda ax: draw_learning_curve(ax, history, turn), "6 · Neural Network")

    stage.tip("`dropout: \"0.5\"`, then `head: \"gap\"`. Which one closes the memorising gap?")
    stage.tip(f"`epochs: \"{turn + 1}\"`: stop where the sit-out loss was lowest. Does the test accuracy agree?")
    stage.tip("`learning_rate` 0.01 and 0.0001: compare the learning curves. Too big jumps around, too small crawls.")
    stage.tip("Beat the forest, or explain why you can't with 360 frames.")
    stage.finish("6 · Neural Network")


def draw_learning_curve(ax, history: dict, turn: int) -> None:
    ax.plot(history["accuracy"], label="frames it learns from")
    ax.plot(history["val_accuracy"], label="frames that sit out")
    ax.axvline(turn, color="#999", ls=":", label=f"sit-out loss lowest (epoch {turn + 1})")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02); ax.legend(fontsize=7)
    ax.set_title("learning curve", fontsize=10)


# ── For a phone: export: tflite | tflite_quantized ───────────────────────────────────────────────────
#    TensorFlow Lite (2017, now LiteRT) flattens the network into one file a phone runs without Python.
#    "quantized" stores every weight as an 8-bit integer instead of a 32-bit float: about 4× smaller.
#    The catch: the network only understands frames that went through stages 1–4 exactly as in training.
#    preprocessing.json writes down what the phone must do first. See how-to/export-a-tflite-file.md.

def export_tflite(model, out_dir, how: str, I_test: np.ndarray, classes: list, cfg: dict) -> np.ndarray:
    """Write model.tflite and preprocessing.json, then grade the FILE on the test set → its predictions."""
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if how == "tflite_quantized":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif how != "tflite":
        raise SystemExit(f"❌ Unknown export '{how}'. Choose none | tflite | tflite_quantized")
    (out_dir / "model.tflite").write_bytes(converter.convert())

    interpreter = tf.lite.Interpreter(model_path=str(out_dir / "model.tflite"))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    preds = []
    for x in I_test:
        interpreter.set_tensor(inp["index"], x[None].astype(inp["dtype"]))
        interpreter.invoke()
        preds.append(int(interpreter.get_tensor(out["index"])[0].argmax()))

    before = ["digital_data", "cleaning", "improving", "segmenting"]
    contract = {"model": "model.tflite", "labels": classes,
                "input": {"shape": [int(v) for v in inp["shape"]], "dtype": np.dtype(inp["dtype"]).name,
                          "values": "0–255 pixel values; the model rescales them itself"},
                "output": "one probability per label, in the order of labels",
                "preprocessing": [{"stage": k, "code": f"{BY_KEY[k].folder}/{BY_KEY[k].code}",
                                   "knobs": {n: v for n, v in cfg[k].items() if not n.startswith("gate_")}}
                                  for k in before]}
    (out_dir / "preprocessing.json").write_text(json.dumps(contract, indent=2, default=str, ensure_ascii=False))
    return np.array(preds)


if __name__ == "__main__":
    run(KEY, main)
