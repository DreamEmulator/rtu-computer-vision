# Todo · Digital Data

## What?

> **Step 1:** Collect underpants. **Step 2:** ? **Step 3:** Profit.
> — the Underpants Gnomes' business plan. Ours has a Step 0, and we actually know what Step 2 is.

### Step 0 · Snap 📸
- [ ] Open **[sdux.tech/computer-vision](https://sdux.tech/computer-vision?repo=__REPO__)**, snap a picture and upload it into the pipeline.
- [ ] First time: the page asks you to connect your repository. After that, every snap runs through your own pipeline.
- [ ] No drip chamber? Snap anything: a glass of water, a bottle, your screen showing a training frame. A real photo against a model trained on synthetic frames is a lesson of its own.
- [ ] Watch the webhooks fire. For **Digital Data** the page shows: your photo as it arrived (resolution, file size) next to what the pipeline really works with, a 128 × 128 grey frame. Under ⚡ performance: milliseconds per frame, and what this runner's CPU offers: SIMD instruction sets and the number of CUDA GPUs (on GitHub's free runners: zero).

<sub>No phone at hand? Run `python run_pipeline.py --snap photo.jpg` locally, or open **Actions → 🚀 CI-Pipeline → Run workflow** and paste an image URL into `snap_url`.</sub>

### Step 1 · Input from the previous stage 📥
Nothing yet: this is the first stage. Its input is the **Natural Scene** itself: the synthetic day-zero dataset (`source: "synthetic"`), your own photos in `data/raw/<class>/` (`source: "folder"`), and your snap.

### Step 2 · Check, improve and play 🎛️
Everything you change lives in the 🎛️ TINKER ZONE of [`digital-data_prepare/action.yaml`](digital-data_prepare/action.yaml). Change one thing, push, and compare the job summary and `preview.png` with the run before.

- [ ] Download the `digital_data` artifact and open `preview.png`: top row the camera frame, bottom row the pipeline's version. Would *you* still spot the drop?
- [ ] `size: "[32, 32]"`: is the drop still there? Nyquist and Shannon say it can't be. Put it back afterwards; every later stage feels this change.
- [ ] At `size: "[64, 64]"`, compare `interpolation: "nearest"` with `"area"`. Look for jagged edges (aliasing).
- [ ] `color_space: "hsv"` with `channel: "2"`, then `"lab"` with `channel: "0"`. Which one looks most like the grey version, and why?
- [ ] Bring your own data: photos in `data/raw/drop/`, `data/raw/no_drop/`, `data/raw/low_fluid/`, then `source: "folder"`.
- [ ] 📓 Go deeper in [`digital-data.ipynb`](digital-data.ipynb).

### Step 3 · Analyse, and decide if we pass on 🚦
- [ ] 🚦 **Gates:** at least 30 frames per class, zero unreadable files.
- [ ] Are the classes balanced? Is the train/test split the same after every run (same `seed`)?
- [ ] Look at your snap's 128 × 128 version on the page. Is the information you need still in there?

**Ready to pass on to Cleaning?** Gates green, and you'd recognise the drop in the pipeline's version yourself → open a pull request: *"Digital Data: what I changed and why"*.

⬅️ [Why this stage exists](README.md) · [The whole pipeline](../README.md)
