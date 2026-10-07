# 🌳 Stage 6 · Classifying · Random Forest

**Lecture:** Image Classification with Random Forests · Tue 06.10 · **The question:** *Is the infusion running, stopped, or about to run dry?*

<p align="center"><img src="pipeline.svg" alt="The Drip Detector CI-pipeline with Classifying · Random Forest highlighted" width="760"></p>

## Why?

📍 **Where we are:** Processing, stage 6: **Classifying**, with a random forest.
On the layer diagram: Natural Scene → Digital Data → Preparing for Analysis → **Model Training** → Inference.

Here the startup's question finally gets an answer: is the infusion running, stopped, or about to run dry?

A random forest learns from the features and is graded only on the test set we locked on day one. In a hospital accuracy alone isn't enough: missing a chamber that runs dry is worse than a false alarm. So we read the confusion matrix and check the recall of every class. Your image for analyzing gets its first real prediction here.

**Without this stage:** all that preparation produces pictures, but no decision.

## How?

This folder is the whole stage:

| File | |
|---|---|
| 📖 `README.md` | you are here: why, and what to try |
| 🎛️ [`action.yaml`](action.yaml) | **this stage's values**: every one you may change, with the mathematician who thought of it, and the 🚦 gates |
| 🐍 [`random_forest.py`](random_forest.py) | **the code CI runs**: the model (`build_model`), the crowd of trees, and how it's evaluated, written to be read top to bottom |

Run it: **▶ `random_forest.py` in PyCharm**, or `python 6_random-forest/random_forest.py`. It runs the stages before it if their output is out of date, then this one. You get, in `build/random_forest/`:

- **`report.md`**: the values you passed in, every number with 🟢 🟠 🔴 and what it means, the gates, what to try next, and **what changed since your last run**
- **`before_after.png`**: the confusion matrix, how the model learned, and the test frames it got wrong

That's exactly what CI shows for this stage when you push, or when you paste a photo URL into **Actions → 🚀 CI-Pipeline → Run workflow**.

## What?

### Step 0 · Image for analyzing 📸
- [ ] Take a photo of what your product should recognise, and put it in [`images_for_analyzing/`](../images_for_analyzing/) (or paste its URL into **Run workflow**).

### Step 1 · Input 📥
What stage 5 passed on, `features.npz`: the training vectors (with augmented copies), the test vectors, the vector of your image for analyzing, and the labels.

### Step 2 · Change a value, run, compare 🎛️
One value at a time in [`action.yaml`](action.yaml), ▶, then read *Since your last run* in the report. Start with the first one.

- [ ] `n_estimators: "5"`, then 200, then 500. Where does the accuracy stop paying for the time? The middle panel of `before_after.png` shows the crowd growing.
- [ ] `max_features: "null"`: every tree may look at every feature. Why does the *wisdom of the crowd* shrink?
- [ ] `max_depth: "3"` and `min_samples_leaf: "10"`: a simpler forest. Worse, or more robust?
- [ ] `class_weight: "balanced"`: does the recall of the worst class move?

### Step 3 · Analyse, and decide if we pass on 🚦
- [ ] 🚦 **Gates:** test accuracy ≥ 0.85, recall of the worst class ≥ 0.75.
- [ ] 🎓 **Cross-validation:** is *cv accuracy* close to *test accuracy*? Set `cv_folds` to 0, 5 and 10: how does the ± change?
- [ ] 🎓 **Confusion matrix:** find the biggest number off the diagonal (left panel of `before_after.png`, and *most common mistake*). Which two classes does it mix up?
- [ ] 🎓 **Precision and recall:** for each class, compare *precision per class* with *recall per class*. Try `class_weight: "balanced"`: which one moves, and what does it cost?
- [ ] Read the confusion matrix. Which mistake is dangerous: calling *low_fluid* "drop", or calling *drop* "low_fluid"? Should that change a gate?
- [ ] Did the forest get your image for analyzing right? If not, what's different between your photo and the training frames? ❓ [How to train our model on cows?](../how-to/train-our-model-on-cows.md)

**Ready to ship?** Gates green, and you can explain the most dangerous mistake. → push, open a pull request: *"Random Forest: what I changed and why"*.

### Going further: change the code
The values tune the forest; [`random_forest.py`](random_forest.py) is where it's built. Swap it in `build_model()` for another scikit-learn classifier, say `GradientBoostingClassifier`, and compare the reports. How it's evaluated stays the same.

⬅️ [The whole pipeline](../README.md)
