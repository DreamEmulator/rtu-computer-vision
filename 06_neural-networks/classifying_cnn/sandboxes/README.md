# 🧪 Sandboxes · Neural Networks

Real pixels in, one idea at a time out. Each sandbox takes the frames stage 4 really passed on (the network learns from pixels, not from stage 5's features) and shows **one** idea from the lecture, with the **same** [`recipe.py`](../recipe.py) and the **same** knobs ([`action.yaml`](../action.yaml)) that CI uses.

```
build/extracting/ ──▶ 6e one filter ──▶ 6f build_network() ──▶ 6g train() ──────▶ 6h what it learned
 stage 4's pixels      by hand             floor by floor          the learning        its filters, what each
                       (no TensorFlow)     (not trained yet)       curve               layer sees, its verdict
```

| | Sandbox | 👾 The Robot… | Recipe | 👀 Look at |
|---|---|---|---|---|
| 6e | [`sandbox_6e_convolution.py`](sandbox_6e_convolution.py) | slides one 3×3 filter over your frame, by hand | (plain OpenCV) | the frame → the filter → ReLU → max pooling |
| 6f | [`sandbox_6f_network.py`](sandbox_6f_network.py) | stacks floors of filters into a network | `build_network()` | numbers and weights per layer |
| 6g | [`sandbox_6g_training.py`](sandbox_6g_training.py) | learns by being wrong, epoch after epoch | `build_network()`, `train()` | accuracy and loss per epoch: learning, or memorising? |
| 6h | [`sandbox_6h_what_it_sees.py`](sandbox_6h_what_it_sees.py) | opens the trained network | (6g's network) | 16 learned filters · what layers 1 and 2 see · the verdict |

The forest's sandboxes are 6a–6d ([`05_random-forests`](../../../05_random-forests/classifying_random-forest/sandboxes/)); the network's continue at 6e. Everything lands in `build/sandbox/`, which is in `.gitignore`.

## How?

**TensorFlow first** (6f–6h need it; 6e doesn't). In PyCharm's Terminal:

```bash
pip install -r requirements-cnn.txt
```

**PyCharm:** the run menu has a folder **🧠 6 · Neural network** with 6e, 6f, 6g, 6h and the recipe. Pick one and press ▶. Keep the `stage_6x_look.png` tab open: it refreshes on every run.

**Terminal**, from the repository root:

```bash
python 06_neural-networks/classifying_cnn/sandboxes/sandbox_6e_convolution.py            # your snap
python 06_neural-networks/classifying_cnn/sandboxes/sandbox_6e_convolution.py low_fluid  # a test frame of that class
python 06_neural-networks/classifying_cnn/sandboxes/sandbox_6f_network.py
python 06_neural-networks/classifying_cnn/sandboxes/sandbox_6g_training.py               # 10–60 seconds
python 06_neural-networks/classifying_cnn/sandboxes/sandbox_6h_what_it_sees.py           # reuses 6g's network
```

The input is always fresh: when `build/extracting/` is missing, or older than a knob or stage you changed in stages 1–5, the sandbox runs the pipeline up to stage 5 first. 6g saves its trained network; 6h reuses it when the knobs and frames still match, and trains a new one otherwise.

## What?

1. **Look first.** Press ▶ with `TRY = {}`: that's exactly the network CI trains. (CI has it switched off until Thursday 08.10: `enabled` in the tinker zone. The sandboxes and ▶ `recipe.py` run it anyway.)
2. **Try one change.** Put one knob in `TRY`, e.g. `TRY = {"head": "gap"}` in 6f, then the same in 6g. A typo stops the run and lists the real knobs.
3. **Another frame** (6e, 6h). `FRAME = "low_fluid"` uses a test frame of that class instead of your snap. 6e also takes your own 3×3 `FILTER`.
4. **Compare.** Open the new `build/sandbox/<unix time>_results_stage_6x.md`: what the Robot did, every knob next to its `action.yaml` value, every metric with 🟢 🟠 🔴, and what to try next. 6f lists every layer; 6h compares every learned filter with Sobel's.
5. **Decide, by hand.** Better than CI? The results file shows the lines to copy into the 🎛️ TINKER ZONE of [`../action.yaml`](../action.yaml). Copying them is your decision (Step 3).
6. **Prove it.** ▶ [`recipe.py`](../recipe.py) (**6 · Network recipe → the exam** in PyCharm): CI's stage with its exam and gates, on your machine. Then push, and let CI prove it (see [`../../Todo_Classifying.md`](../../Todo_Classifying.md)).

## Going further: change the recipe itself

Knobs reshape the network somebody already designed. [`recipe.py`](../recipe.py) is where it's built and trained: add a layer, replace `MaxPooling2D` with a convolution with `strides=2`, try `tf.keras.optimizers.SGD`, or stop at the best epoch with `tf.keras.callbacks.EarlyStopping`. 6f shows the new shape at once, 6g whether it learns, and CI's gates whether it finally beats the forest.

⬅️ [The action](../action.yaml) · [Why this stage exists](../../README.md) · [The whole pipeline](../../../README.md)
