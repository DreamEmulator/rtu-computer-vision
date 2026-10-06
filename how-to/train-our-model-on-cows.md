# ❓ How to train our model on cows?

> 🎯 **Goal** · a model that recognises cows on photos it has never seen, and an explanation of why each change helped.<br>
> 🗺️ **Route** · 🌳 6 read the mistakes → back to the stage that caused them → 🌳 6 measure again<br>
> 🧰 **You need** · your classes in the pipeline first: [How to add my own classes?](add-my-own-classes.md)

Making a model better isn't turning values until the number goes up. It's detective work: look at the mistakes, guess the cause, change **one** thing, measure. This How To is that loop.

## 🪜 Steps

### 1 · Write down the honest baseline
- [ ] `group_by: "prefix"` in stage 1, then ▶ **Pipeline · everything**.
- [ ] From the 🌳 Random Forest output, note: `test accuracy`, the cow entry of `recall per class`, and `baseline (always guess the majority)`.
- [ ] Take three *new* cow photos, not from your videos, and send each through: `python run_pipeline.py --snap cow1.jpg --snap-label cow`. How many does it get right?

The gap between the test accuracy and your three photos is how far your test set is from the real world.

### 2 · Look at the mistakes
- [ ] Open `build/random_forest/before_after.png`: the confusion matrix on the left, the misclassified test frames on the right, each labelled `true → predicted`.
- [ ] For every wrong frame, write one line: *what in this frame could have fooled the model?*
- [ ] Sort those lines into piles. The biggest pile is your next experiment.

### 3 · Follow the fork of your biggest pile (below)

### 4 · One change, one pull request
- [ ] A branch per experiment, one value per branch. Fill in the PR template's hypothesis *before* you run, and the numbers after.
- [ ] Didn't help? Close the PR and write down why. A negative result is still a result.

### 5 · Back to step 1
Measure again, look at the new mistakes, pick the next pile.

## 🔀 Which pile is biggest?

| The mistakes are… | It means | Try |
|---|---|---|
| 🌱 cows on unusual backgrounds; horses on grass called *cow* | the model learned the **grass**, not the cow | film cows somewhere else and horses on grass; make segmentation cut the background away |
| ✂️ half a cow, or no cow at all, in `build/segmenting/before_after.png` | **segmentation eats the cow**: the threshold was made for a dark chamber on a bright wall | [stage 4](../4_segmenting/README.md): `invert`, `threshold: "otsu"`, a bigger `min_contour_area`; ▶ `segmenting.py` and read *Since your last run* |
| 🎨 black-and-white cows and brown horses mixed up | the model is **colour-blind**: stage 1 makes everything grey, and stage 5 only looks at brightness | `color_space: "lab"` with `channel: 1` (green ↔ red): grass goes dark, brown goes light, black and white stay in the middle. Or the 🧠 network, which sees every channel |
| 🔍 cows far away | at 128 × 128 a cow 30 px wide in a 640 px frame becomes 6 px wide: Nyquist says it's gone | `size: "[256, 256]"`, or `crop` to where the cows are |
| ⚖️ cow recall low while the other classes are fine | too few cows, or too little variety | `class_weight: "balanced"`, `augment_copies: "3"`, another video |
| 🤷 no pattern, and every change moves the score by ±1 % | the features are the ceiling, not the values | [How to pick which model I am training?](pick-which-model-i-am-training.md) |

Two tests that tell you more than any value:

- **The background test.** Show it a cow photo on your laptop screen in front of a white wall. Still a cow? Then it learned the cow.
- **The ablation.** `pass_on: "original"` in Segmenting keeps the stage but ignores its result. If accuracy goes *up*, segmentation was hurting: fix it, or honestly leave it out.

## 🏆 Done when
- Cow recall clears the gate and the target you set in step 1.
- Three new cow photos right, and three non-cow photos not called *cow*.
- You can name the one change that mattered most, and why.

## 💼 At work this is called…
- **Error analysis**: look at the mistakes before you change anything. Andrew Ng's *Machine Learning Yearning* (2018) recommends doing it by hand, on about a hundred examples.
- **Shortcut learning** (Geirhos et al., 2020): a model takes the easiest clue in the data. The classic is the "husky vs wolf" classifier that turned out to be a snow detector (Ribeiro, Singh & Guestrin, 2016).
- **Ablation**: switch one part off to measure what it contributes.
- **Domain shift**: the test set and the real world look different.
- **Data-centric AI**: improve the data, not the model.

⬅️ [All How To's](README.md) · [The whole pipeline](../README.md)
