# Todo · Classifying · Random Forest

## What?

> **Step 1:** Collect underpants. **Step 2:** ? **Step 3:** Profit.
> — the Underpants Gnomes' business plan. Ours has a Step 0, and we actually know what Step 2 is.

### Step 0 · Snap 📸
- [ ] Open **[sdux.tech/computer-vision](https://sdux.tech/computer-vision?repo=DreamEmulator/rtu-computer-vision)**, snap a picture and upload it into the pipeline.
- [ ] Watch the webhooks fire. For **Classifying · Random Forest** the page shows: the forest's verdict on your snap (drop, no_drop or low_fluid), how sure it is, and the probability of every class. If you told the page what's on the photo, a ✓ or ✗. Performance: training seconds, milliseconds per prediction and the model's size.

<sub>No phone at hand? Run `python run_pipeline.py --snap photo.jpg` locally, or open **Actions → 🚀 CI-Pipeline → Run workflow** and paste an image URL into `snap_url`.</sub>

### Step 1 · Input from the previous stage 📥
The `extracting` artifact, `features.npz`: the training vectors (with augmented copies), the test vectors, your snap's vector, and the labels.

### Step 2 · Check, improve and play 🎛️
Everything you change lives in the 🎛️ TINKER ZONE of [`classifying_random-forest/action.yaml`](classifying_random-forest/action.yaml). Change one thing, push, and compare the job summary and `preview.png` with the run before.

- [ ] 🧪 Follow the lecture in the [sandboxes](classifying_random-forest/sandboxes/), ▶ in PyCharm: **6a** one tree and the questions it asks your snap, **6b** a forest of them, **6c** how they vote on your snap and where they look, **6d** the exam. Same [`recipe.py`](classifying_random-forest/recipe.py) as CI. Put one knob in `TRY`, compare the results files, then copy the winner into the tinker zone yourself.
- [ ] 6b: `TRY = {"max_features": None}`. Why does a forest of trees that may look at *everything* do worse?


- [ ] `n_estimators: "5"`, then 200, then 500. Plot training seconds against accuracy. Where does it stop paying off?
- [ ] `max_depth: "3"` and `min_samples_leaf: "10"`: a simpler forest. Worse, or more robust?
- [ ] `model: "svm"` and `"knn"`. Then switch off `scaling` in stage 5 and run both again. The whole comparison: ❓ [How to pick which model I am training?](../how-to/pick-which-model-i-am-training.md)
- [ ] Compare the **cv accuracy** with the **test accuracy**. In the notebook, see what leaky cross-validation (augmented copies in both folds) would have told you.
- [ ] `class_weight: "balanced"`: does the recall of the worst class move?
- [ ] 📓 Which HOG cells the forest actually uses in [`random-forest.ipynb`](random-forest.ipynb).

### Step 3 · Analyse, and decide if we pass on 🚦
- [ ] 🚦 **Gates:** test accuracy ≥ 0.85, recall of the worst class ≥ 0.75.
- [ ] Read the confusion matrix. Which mistake is dangerous: calling *low_fluid* "drop", or calling *drop* "low_fluid"? Should that change a gate?
- [ ] Did the forest get your snap right? If not: what's different between your photo and the training frames? ❓ [How to train our model on cows?](../how-to/train-our-model-on-cows.md) shows how to find out.

**Ready to ship?** Gates green, and you can explain the most dangerous mistake → pull request: *"Random forest: what I changed and why"*.

⬅️ [Why this stage exists](README.md) · [The whole pipeline](../README.md)
