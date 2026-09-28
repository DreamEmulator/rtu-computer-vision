# Todo · Segmenting

## What?

> **Step 1:** Collect underpants. **Step 2:** ? **Step 3:** Profit.
> — the Underpants Gnomes' business plan. Ours has a Step 0, and we actually know what Step 2 is.

### Step 0 · Snap 📸
- [ ] Open **[sdux.tech/computer-vision](https://sdux.tech/computer-vision?repo=__REPO__)**, snap a picture and upload it into the pipeline.
- [ ] Watch the webhooks fire. For **Segmenting** the page shows: your snap with the mask painted green and every contour in red, the share of your photo that counts as foreground, and the number of contours.

<sub>No phone at hand? Run `python run_pipeline.py --snap photo.jpg` locally, or open **Actions → 🚀 CI-Pipeline → Run workflow** and paste an image URL into `snap_url`.</sub>

### Step 1 · Input from the previous stage 📥
The `improving` artifact: enhanced frames with decent contrast, even from the night shift.

### Step 2 · Check, improve and play 🎛️
Everything you change lives in the 🎛️ TINKER ZONE of [`segmenting_threshold/action.yaml`](segmenting_threshold/action.yaml). Change one thing, push, and compare the job summary and `preview.png` with the run before.

- [ ] `threshold: "otsu"` versus `"adaptive"`. Look at the overlay: what does one global threshold do with uneven light?
- [ ] Morphology: `"[open]"`, `"[close]"`, `"[open, close]"`, and `morph_kernel: "5"`. Which removes specks, which closes gaps?
- [ ] `min_contour_area: "150"`: the specks disappear… and so might the drop. It's small.
- [ ] `fill_holes: "false"`: outlines only. What changes for the features?
- [ ] `pass_on`: `crop` zooms in on the biggest object, `mask` passes only the shape, `original` ignores the segmentation. Compare the final accuracy of each.
- [ ] 📓 Otsu's threshold on the histogram in [`segmenting.ipynb`](segmenting.ipynb).

### Step 3 · Analyse, and decide if we pass on 🚦
- [ ] 🚦 **Gates:** at most 5 % of frames with an empty mask, at most 5 % with a full one.
- [ ] Is the chamber in *your* snap segmented, or the wall behind it?

**Ready to pass on to Extracting?** Gates green and the overlays hug the chamber → pull request: *"Segmenting: what I changed and why"*.

⬅️ [Why this stage exists](README.md) · [The whole pipeline](../README.md)
