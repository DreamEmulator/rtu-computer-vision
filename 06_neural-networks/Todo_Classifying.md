# Todo · Classifying · Neural Network

## What?

> **Step 1:** Collect underpants. **Step 2:** ? **Step 3:** Profit.
> — the Underpants Gnomes' business plan. Ours has a Step 0, and we actually know what Step 2 is.

### Step 0 · Snap 📸
- [ ] Open **[sdux.tech/computer-vision](https://sdux.tech/computer-vision?repo=DreamEmulator/rtu-computer-vision)**, snap a picture and upload it into the pipeline.
- [ ] Watch the webhooks fire. For **Classifying · Neural Network** the page shows: the network's verdict on your snap next to the forest's. Do they agree? Training seconds and milliseconds per prediction, side by side with the forest.

<sub>No phone at hand? Run `python run_pipeline.py --snap photo.jpg` locally, or open **Actions → 🚀 CI-Pipeline → Run workflow** and paste an image URL into `snap_url`.</sub>

### Step 1 · Input from the previous stage 📥
Also the `extracting` artifact, but a different part of it: the CNN ignores the feature vectors and learns from the segmented **pixels** (`I_train`, `I_test`, `I_snap`).

### Step 2 · Check, improve and play 🎛️
Everything you change lives in the 🎛️ TINKER ZONE of [`classifying_cnn/action.yaml`](classifying_cnn/action.yaml). Change one thing, push, and compare the job summary and `preview.png` with the run before.

- [ ] `enabled: "true"`, push, and watch it fail the accuracy gate. Open the learning curve in `preview.png`: training accuracy near 100 %, validation far below. That's overfitting.
- [ ] Fight it: `dropout: "0.5"`, fewer `epochs`, or more data with `augment_copies: "3"` in stage 5 (the classic cure).
- [ ] `batch_norm: "true"`: does it help here? Why might it struggle with so few frames?
- [ ] `input_downscale` at 1 and 4, `head: "gap"` versus `"flatten"`.
- [ ] Compare training seconds with the forest. Now imagine an NVIDIA GPU and CUDA: this is why deep learning waited until 2012.
- [ ] 📓 Filters and learning curves in [`neural-network.ipynb`](neural-network.ipynb).

### Step 3 · Analyse, and decide if we pass on 🚦
- [ ] 🚦 **Gates:** the same promises as the forest: accuracy ≥ 0.85, worst-class recall ≥ 0.75.
- [ ] Beat the forest, or explain why you can't with 360 frames.

**Ready to ship?** A model that doesn't beat the forest doesn't ship, however modern it is. Keep `enabled: "true"` only when its gates pass, and argue your case in the pull request: *"Neural network: what I changed and why"*.

⬅️ [Why this stage exists](README.md) · [The whole pipeline](../README.md)
