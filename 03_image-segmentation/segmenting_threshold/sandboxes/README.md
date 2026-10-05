# 🧪 Sandboxes · Image Segmentation

One real frame in, one image out. Each sandbox runs **one** sub-step of stage 4 · Segmenting on **one** frame, with the **same** [`recipe.py`](../recipe.py) and the **same** knobs ([`action.yaml`](../action.yaml)) that CI uses on every frame. What you see here is what the pipeline does.

```
build/improving/  ──▶ 4a threshold() ──▶ 4b morphology() ──▶ 4c keep_contours() ──▶ 4d boundary() + apply() ──▶ stage 5
 your snap, as         mask               repaired mask        the objects we keep    crust · what Extracting
 stage 3 passed it on                                                                    receives
```

| | Sandbox | 👾 The Robot… | Recipe | 📥 Reads | 👀 Look at |
|---|---|---|---|---|---|
| 4a | [`sandbox_4a_threshold.py`](sandbox_4a_threshold.py) | sorts pixels into a dark and a light bin | `threshold()` | your snap from `build/improving/` | `stage_4a_look.png` |
| 4b | [`sandbox_4b_morphology.py`](sandbox_4b_morphology.py) | planes and fills the mask like a woodworker | `morphology()` | `stage_4a_threshold.png` | `stage_4b_look.png` |
| 4c | [`sandbox_4c_contours.py`](sandbox_4c_contours.py) | traces every blob like your hand on paper | `keep_contours()` | `stage_4b_morphology.png` | `stage_4c_look.png` |
| 4d | [`sandbox_4d_boundary.py`](sandbox_4d_boundary.py) | scoops out the cookie, keeps the crust | `boundary()`, `apply()` | `stage_4c_contours.png` | `stage_4d_look.png` |

Everything lands in `build/sandbox/`. That folder is in `.gitignore`, so your experiments stay on your machine.

## How?

**PyCharm:** the run menu (top right, next to ▶) already has a folder **✂️ 4 · Segmenting** with 4a, 4b, 4c, 4d and the recipe. Pick one and press ▶. Open the `stage_4x_look.png` it prints once and keep its tab open: it refreshes on every run.

**Terminal**, from the repository root, after `pip install -r requirements.txt`:

```bash
python 03_image-segmentation/segmenting_threshold/sandboxes/sandbox_4a_threshold.py              # your snap
python 03_image-segmentation/segmenting_threshold/sandboxes/sandbox_4a_threshold.py my_photo.jpg # any photo, as it is
python 03_image-segmentation/segmenting_threshold/sandboxes/sandbox_4b_morphology.py             # picks up 4a's output
python 03_image-segmentation/segmenting_threshold/sandboxes/sandbox_4c_contours.py
python 03_image-segmentation/segmenting_threshold/sandboxes/sandbox_4d_boundary.py
```

The input is always fresh. When `build/improving/` is missing, or older than a knob or stage you changed in stages 1–3, sandbox 4a runs the pipeline up to stage 3 first. Change Improving's tinker zone and 4a shows you the knock-on effect.

## What?

1. **Look first.** Press ▶ with `TRY = {}`: that's exactly what CI does to your snap.
2. **Try one change.** Put one knob in `TRY`, e.g. `TRY = {"threshold": "otsu"}`, and press ▶ again. Run the sandboxes after it too, if you want the knock-on effect. A typo stops the run and lists the real knobs.
3. **Another frame.** `FRAME = "low_fluid"` (in 4a) runs the same knobs on a test frame of that class. One rule has to fit every frame, not just yours.
4. **Compare.** Open the new `build/sandbox/<unix time>_results_stage_4x.md`. It says what the Robot did, every knob next to its `action.yaml` value, every metric with 🟢 🟠 🔴 and what to try next. The unix time sorts your runs oldest to newest.
5. **Decide, by hand.** Better than CI? The results file shows the lines to copy into the 🎛️ TINKER ZONE of [`../action.yaml`](../action.yaml). Copying them is your decision (Step 3), so no script does it for you.
6. **Prove it on every frame.** ▶ [`recipe.py`](../recipe.py) (**4 · Recipe on every frame → accuracy** in PyCharm). It runs your recipe on all frames, then Extracting and the random forest, in about 5 seconds. Did the test accuracy go up? Push, and let CI prove it (see [`../../Todo_Segmenting.md`](../../Todo_Segmenting.md)).

## Going further: change the recipe itself

Knobs choose between methods somebody already wrote. [`recipe.py`](../recipe.py) is where the methods live, and CI runs it as it is. Add a method of your own, say a `"kmeans"` brain in `threshold()`, and pick it with `TRY = {"threshold": "kmeans"}`. The sandboxes call it at once, CI calls it on the next push, and the 🚦 gates decide whether it's good enough to ship.

⬅️ [The action](../action.yaml) · [Why this stage exists](../../README.md) · [The whole pipeline](../../../README.md)
