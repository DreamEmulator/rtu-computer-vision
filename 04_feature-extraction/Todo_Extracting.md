# Todo · Extracting

## What?

> **Step 1:** Collect underpants. **Step 2:** ? **Step 3:** Profit.
> — the Underpants Gnomes' business plan. Ours has a Step 0, and we actually know what Step 2 is.

### Step 0 · Snap 📸
- [ ] Open **[sdux.tech/computer-vision](https://sdux.tech/computer-vision?repo=DreamEmulator/rtu-computer-vision)**, snap a picture and upload it into the pipeline.
- [ ] Watch the webhooks fire. For **Extracting** the page shows: the HOG picture of your snap (which way its edges point) and its LBP texture codes, the length of its feature vector, and milliseconds per frame.

<sub>No phone at hand? Run `python run_pipeline.py --snap photo.jpg` locally, or open **Actions → 🚀 CI-Pipeline → Run workflow** and paste an image URL into `snap_url`.</sub>

### Step 1 · Input from the previous stage 📥
The `segmenting` artifact: frames with the background masked away.

### Step 2 · Check, improve and play 🎛️
Everything you change lives in the 🎛️ TINKER ZONE of [`extracting_describe/action.yaml`](extracting_describe/action.yaml). Change one thing, push, and compare the job summary and `preview.png` with the run before.

- [ ] `features: "[lbp]"` only, then `"[hog]"` only, then add `histogram` or `pixels`. Which feature carries the drop?
- [ ] `hog_pixels_per_cell`: 32, 8 and 4. Watch the vector length, the gate (the camera chip's memory) and the accuracy.
- [ ] `augment_copies`: 0 versus 3. More data, or just more of the same?
- [ ] `augment_flip_vertical: "true"`: drops don't fall upwards. What happens to the accuracy?
- [ ] `scaling: "none"`: the forest won't care (why not?). Try `model: "svm"` or `"knn"` in `05_random-forests` afterwards and watch them suffer.
- [ ] 📓 HOG cells and LBP codes up close in [`extracting.ipynb`](extracting.ipynb).

### Step 3 · Analyse, and decide if we pass on 🚦
- [ ] 🚦 **Gates:** feature vector ≤ 20 000 numbers, and no NaN values.
- [ ] Check that augmentation touched the train split only: the test count in the summary must not change.

**Ready to pass on to Classifying?** Gates green, vector small enough for the chip → pull request: *"Extracting: what I changed and why"*.

⬅️ [Why this stage exists](README.md) · [The whole pipeline](../README.md)
