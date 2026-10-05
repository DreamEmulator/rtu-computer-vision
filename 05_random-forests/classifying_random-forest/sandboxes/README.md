# 🧪 Sandboxes · Random Forests

The real features in, one idea at a time out. Each sandbox takes stage 5's real feature vectors (and your snap's) and shows **one** idea from the lecture, with the **same** [`recipe.py`](../recipe.py) and the **same** knobs ([`action.yaml`](../action.yaml)) that CI uses. What you see here is the forest CI trains.

```
build/extracting/  ──▶ 6a grow_tree() ──▶ 6b build_model() ──▶ 6c votes() ──────▶ 6d the exam 🔒
 features.npz          one tree,            200 trees,           how they vote       the locked test set
                       20 questions         one crowd            on your frame       and the gates
```

| | Sandbox | 👾 The Robot… | Recipe | 👀 Look at |
|---|---|---|---|---|
| 6a | [`sandbox_6a_tree.py`](sandbox_6a_tree.py) | plays twenty questions with your frame | `grow_tree()` | the questions it asked, boxed on your frame · the top of the tree |
| 6b | [`sandbox_6b_forest.py`](sandbox_6b_forest.py) | asks a crowd instead of one expert (Galton's ox) | `build_model()` | every lone tree's score vs the forest's · the score as trees join |
| 6c | [`sandbox_6c_verdict.py`](sandbox_6c_verdict.py) | holds an election on your frame | `votes()` | where in the frame the forest looks · the votes |
| 6d | [`sandbox_6d_exam.py`](sandbox_6d_exam.py) | sits CI's exam on frames it has never seen | 🔒 `grade()` | confusion matrix · recall per class · the frames it got wrong |

The exam (6d) lives in 🔒 `stages/s6_classifying.py`, not in the recipe, on purpose: you change the model, never the exam.

Everything lands in `build/sandbox/`. That folder is in `.gitignore`, so your experiments stay on your machine.

## How?

**PyCharm:** the run menu has a folder **🌳 6 · Random forest** with 6a, 6b, 6c, 6d and the recipe. Pick one and press ▶. Keep the `stage_6x_look.png` tab open: it refreshes on every run.

**Terminal**, from the repository root:

```bash
python 05_random-forests/classifying_random-forest/sandboxes/sandbox_6a_tree.py             # your snap
python 05_random-forests/classifying_random-forest/sandboxes/sandbox_6a_tree.py low_fluid   # a test frame of that class
python 05_random-forests/classifying_random-forest/sandboxes/sandbox_6b_forest.py
python 05_random-forests/classifying_random-forest/sandboxes/sandbox_6c_verdict.py
python 05_random-forests/classifying_random-forest/sandboxes/sandbox_6d_exam.py
```

The input is always fresh. When `build/extracting/` is missing, or older than a knob or stage you changed in stages 1–5, the sandbox runs the pipeline up to stage 5 first. Change a segmentation knob and 6c shows you whether the forest now looks somewhere else.

## What?

1. **Look first.** Press ▶ with `TRY = {}`: that's exactly the model CI trains.
2. **Try one change.** Put one knob in `TRY`, e.g. `TRY = {"max_features": None}` in 6b, and press ▶ again. A typo stops the run and lists the real knobs.
3. **Another frame** (6a, 6c). `FRAME = "low_fluid"` asks about a test frame of that class instead of your snap.
4. **Compare.** Open the new `build/sandbox/<unix time>_results_stage_6x.md`: what the Robot did, every knob next to its `action.yaml` value, every metric with 🟢 🟠 🔴, and what to try next. 6a also writes out the tree's rules; 6c lists the features the forest relies on most.
5. **Decide, by hand.** Better than CI? The results file shows the lines to copy into the 🎛️ TINKER ZONE of [`../action.yaml`](../action.yaml). Copying them is your decision (Step 3).
6. **Prove it.** ▶ [`recipe.py`](../recipe.py) (**6 · Recipe → the exam** in PyCharm): CI's stage, on your machine, in a few seconds. Then push, and let CI prove it (see [`../../Todo_Classifying.md`](../../Todo_Classifying.md)).

## Going further: change the recipe itself

Knobs choose between models somebody already wired up. [`recipe.py`](../recipe.py) is where they're built. Add one of your own to `build_model()`, say scikit-learn's `GradientBoostingClassifier` as `"gradient_boosting"`, and pick it with `TRY = {"model": "gradient_boosting"}`. 6b and 6d run it at once, CI on the next push, and the 🚦 gates decide whether it ships.

⬅️ [The action](../action.yaml) · [Why this stage exists](../../README.md) · [The whole pipeline](../../../README.md)
