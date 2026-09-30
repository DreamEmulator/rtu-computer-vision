# Todo · Improving

## What?

> **Step 1:** Collect underpants. **Step 2:** ? **Step 3:** Profit.
> — the Underpants Gnomes' business plan. Ours has a Step 0, and we actually know what Step 2 is.

### Step 0 · Snap 📸
- [ ] Open **[sdux.tech/computer-vision](https://sdux.tech/computer-vision?repo=DreamEmulator/rtu-computer-vision)**, snap a picture and upload it into the pipeline.
- [ ] Watch the webhooks fire. For **Improving** the page shows: your snap enhanced, and its Canny edges. Contrast of your photo before and after, the share of edge pixels, and milliseconds per frame for both the enhancement and Canny.

<sub>No phone at hand? Run `python run_pipeline.py --snap photo.jpg` locally, or open **Actions → 🚀 CI-Pipeline → Run workflow** and paste an image URL into `snap_url`.</sub>

### Step 1 · Input from the previous stage 📥
The `cleaning` artifact: denoised frames. Almost half of them were shot on the night shift: dim, with every value squeezed into a narrow band of grey.

### Step 2 · Check, improve and play 🎛️
Everything you change lives in the 🎛️ TINKER ZONE of [`improving_enhance/action.yaml`](improving_enhance/action.yaml). Change one thing, push, and compare the job summary and `preview.png` with the run before.

- [ ] `method: "none"`: the dim-contrast gate fails… but check Demo Day: the accuracy may go *up*. HOG normalises contrast per block (stage 5). So who is right, the gate or the accuracy? Bring your answer to class.
- [ ] `equalize` versus `clahe` on the dim frames. Then `clahe_clip` at 1, 2 and 4: where does noise start to bloom?
- [ ] `method: "gamma"` with `gamma: "0.5"`: brighter shadows, but is it better?
- [ ] Canny at `canny_low: "20"`, `canny_high: "60"` versus `100` / `200`. Then `pass_on: "edges"` and follow the consequences downstream.
- [ ] 📓 Histograms before and after in [`improving.ipynb`](improving.ipynb).

### Step 3 · Analyse, and decide if we pass on 🚦
- [ ] 🚦 **Gate:** brightness σ of the dimmest 5 % of frames ≥ 15.
- [ ] Look at the dimmest frames in `preview.png`. Would a nurse see anything on the night shift?

**Ready to pass on to Segmenting?** Gate green and the night frames readable → pull request: *"Improving: what I changed and why"*.

⬅️ [Why this stage exists](README.md) · [The whole pipeline](../README.md)
