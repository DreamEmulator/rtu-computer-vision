# 🧪 Sandboxes · Feature Extraction and Data Preparation

Real frames in, numbers out, one idea at a time. Each sandbox runs **one** part of stage 5 · Extracting on the frames stage 4 really passed on, with the **same** [`recipe.py`](../recipe.py) and the **same** knobs ([`action.yaml`](../action.yaml)) that CI uses.

| Extraction | Scaling & Normalization | Data Augmentation |
|---|---|---|
| **5a** Histogram of Oriented Gradients · **5b** Local Binary Patterns | **5c** Min-Max Scaling · Z-Score Standardization | **5d** Rotation · Flipping · Scaling |

```
build/segmenting/ ──▶ 5a hog_features()  ─┐
 your snap and         which way the       ├─▶ 5c fit_scaler() ──▶ stage 6
 every frame           edges point          │    one ruler for every feature,
                    ──▶ 5b lbp_features() ─┘    measured on the training frames
                        what the texture is

In CI, 5d augment() runs first: the training frames get copies, then everything is described and scaled.
```

| | Sandbox | 👾 The Robot… | Recipe | 👀 Look at |
|---|---|---|---|---|
| 5a | [`sandbox_5a_hog.py`](sandbox_5a_hog.py) | asks every cell which way its edges point | `hog_features()` | the cells · the edges · HOG's stars · the strongest cell's directions |
| 5b | [`sandbox_5b_lbp.py`](sandbox_5b_lbp.py) | gives every pixel a texture code from its neighbours | `lbp_codes()`, `lbp_features()` | the grid · a code per pixel · the codes counted |
| 5c | [`sandbox_5c_scaling.py`](sandbox_5c_scaling.py) | puts centimetres and kilometres on one ruler | `fit_scaler()` | three features before and after · kNN and the forest, scaled and not |
| 5d | [`sandbox_5d_augmentation.py`](sandbox_5d_augmentation.py) | makes plausible new camera frames from old ones | `flip()`, `affine()`, `brightness()`, `augment()` | every effect at its limit · random copies like CI makes |

Everything lands in `build/sandbox/`. That folder is in `.gitignore`, so your experiments stay on your machine.

## How?

**PyCharm:** the run menu has a folder **🧬 5 · Extracting** with 5a, 5b, 5c, 5d and the recipe. Pick one and press ▶. Keep the `stage_5x_look.png` tab open: it refreshes on every run.

**Terminal**, from the repository root:

```bash
python 04_feature-extraction/extracting_describe/sandboxes/sandbox_5a_hog.py             # your snap
python 04_feature-extraction/extracting_describe/sandboxes/sandbox_5a_hog.py low_fluid   # a test frame of that class
python 04_feature-extraction/extracting_describe/sandboxes/sandbox_5b_lbp.py
python 04_feature-extraction/extracting_describe/sandboxes/sandbox_5c_scaling.py         # every frame
python 04_feature-extraction/extracting_describe/sandboxes/sandbox_5d_augmentation.py
```

The input is always fresh. When `build/segmenting/` is missing, or older than a knob or stage you changed in stages 1–4, the sandbox runs the pipeline up to stage 4 first.

## What?

1. **Look first.** Press ▶ with `TRY = {}`: that's exactly what CI does.
2. **Try one change.** Put one knob in `TRY`, e.g. `TRY = {"hog_pixels_per_cell": 8}` in 5a, and press ▶ again. A typo stops the run and lists the real knobs.
3. **Another frame** (5a, 5b, 5d). `FRAME = "low_fluid"` uses a test frame of that class instead of your snap.
4. **Compare.** Open the new `build/sandbox/<unix time>_results_stage_5x.md`: what the Robot did, every knob next to its `action.yaml` value, every metric with 🟢 🟠 🔴, and what to try next.
5. **Decide, by hand.** Better than CI? The results file shows the lines to copy into the 🎛️ TINKER ZONE of [`../action.yaml`](../action.yaml). Copying them is your decision (Step 3).
6. **Prove it.** ▶ [`recipe.py`](../recipe.py) (**5 · Recipe → the exam** in PyCharm): your features on every frame, then the random forest's exam, in a few seconds. Then push, and let CI prove it (see [`../../Todo_Extracting.md`](../../Todo_Extracting.md)).

## Going further: change the recipe itself

Knobs choose between features somebody already wrote. [`recipe.py`](../recipe.py) is where they're computed. Add one of your own to `extract()`, say `"edges"`: the share of Canny edge pixels per cell of a grid. Pick it with `TRY = {"features": ["hog", "lbp", "edges"]}`; 5c scales it with the rest, and ▶ `recipe.py` tells you whether the forest found it worth its numbers.

⬅️ [The action](../action.yaml) · [Why this stage exists](../../README.md) · [The whole pipeline](../../../README.md)
