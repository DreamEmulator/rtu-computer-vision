"""
🧠 Stage 6 · Classifying — the neural network recipe
════════════════════════════════════════════════════
The network and how it learns, nothing else. CI runs exactly this file:

    .github/workflows/pipeline.yml
      └─ classifying_cnn/action.yaml       🎛️ knobs + 🚦 gates
           └─ stages/s6_classifying.py      🔒 plumbing: load the pixels, the exam, export, gates
                └─ recipe.py                🧪 you are here

The sandboxes next to this file:

    6e one filter by hand → 6f build_network() → 6g train() → 6h what it learned

Every function gets `knobs`: the TINKER ZONE of action.yaml (plus a sandbox's TRY).
Change a knob there; change the *architecture* here. Add a layer, swap MaxPooling for strides,
try another optimizer, and let the gates tell you whether it beats the forest.

▶ Press ▶ on this file: your network trains on every frame and takes the exam (needs TensorFlow:
  pip install -r requirements-cnn.txt).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

layers = tf.keras.layers


# ── 6f · the architecture ──── 👾 stacks of filters that it will learn itself, then a vote

def build_network(input_shape: tuple, n_classes: int, knobs: dict, seed: int) -> tf.keras.Sequential:
    """A small convolutional network (LeCun et al., 1989): convolution → ReLU → pooling, a few times, then a head."""
    tf.keras.utils.set_random_seed(seed)                       # same seed, same starting weights
    model = tf.keras.Sequential([layers.Input(shape=input_shape), layers.Rescaling(1 / 255.0)])
    if (knobs.get("input_downscale") or 1) > 1:
        model.add(layers.AveragePooling2D(knobs["input_downscale"]))   # fewer pixels: faster, coarser
    for i in range(knobs.get("conv_layers", 3)):
        model.add(layers.Conv2D(knobs.get("filters", 16) * 2 ** i, 3, padding="same", use_bias=not knobs.get("batch_norm")))
        if knobs.get("batch_norm"):
            model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))                   # keep what fired, drop the rest
        model.add(layers.MaxPooling2D())                       # half the width and height, keep the strongest
    model.add(layers.GlobalAveragePooling2D() if knobs.get("head", "flatten") == "gap" else layers.Flatten())
    model.add(layers.Dense(knobs.get("dense_units", 64), activation="relu"))
    model.add(layers.Dropout(knobs.get("dropout", 0.3)))
    model.add(layers.Dense(n_classes, activation="softmax"))   # one probability per class
    return model


# ── 6g · training ──────────── 👾 guess, measure how wrong, nudge every weight downhill, repeat

def train(model: tf.keras.Sequential, I_train: np.ndarray, y_train: np.ndarray, knobs: dict, seed: int,
          verbose: int = 2) -> dict:
    """Gradient descent with Adam (Kingma & Ba, 2014). 15 % of the training frames are held back to watch for overfitting.

    Returns the history: accuracy and loss per epoch, on the frames it trains on and on the held-back ones.
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(knobs.get("learning_rate", 1e-3)),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    order = np.random.default_rng(seed).permutation(len(y_train))      # mix the classes before the hold-back split
    history = model.fit(I_train[order], y_train[order], validation_split=0.15, epochs=knobs.get("epochs", 40),
                        batch_size=knobs.get("batch_size", 32), verbose=verbose)
    return history.history


if __name__ == "__main__":  # ▶ this network on every frame, then the exam
    import json
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run_pipeline import run
    from stages.common import knobs_of
    from stages.sandbox import refresh

    refresh("extracting")                                   # stages 1–5 again, if you changed one of them
    if not knobs_of("neural_network").get("enabled"):
        print("👾 enabled is false in action.yaml, so CI skips the network. I'm switching it on for this run only.")
        os.environ.update(CV_STAGE="neural_network", CV_KNOBS=json.dumps({"enabled": "true"}))
    ok = run("neural_network", "neural_network", keep_going=True)
    print("\n👀 The exam and the learning curve: build/neural_network/preview.png")
    raise SystemExit(0 if ok else 1)
