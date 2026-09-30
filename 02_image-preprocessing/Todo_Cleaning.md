# Todo · Cleaning

## What?

> **Step 1:** Collect underpants. **Step 2:** ? **Step 3:** Profit.
> — the Underpants Gnomes' business plan. Ours has a Step 0, and we actually know what Step 2 is.

### Step 0 · Snap 📸
- [ ] Open **[sdux.tech/computer-vision](https://sdux.tech/computer-vision?repo=DreamEmulator/rtu-computer-vision)**, snap a picture and upload it into the pipeline.
- [ ] Watch the webhooks fire. For **Cleaning** the page shows: your snap before and after the filter, plus **removed ×4**: exactly what the filter took away, amplified. The noise σ of your photo before and after, and milliseconds per frame.

<sub>No phone at hand? Run `python run_pipeline.py --snap photo.jpg` locally, or open **Actions → 🚀 CI-Pipeline → Run workflow** and paste an image URL into `snap_url`.</sub>

### Step 1 · Input from the previous stage 📥
The `digital_data` artifact: 128 × 128 grey frames. About a third of them carry camera damage: Gaussian sensor noise, salt-and-pepper dead pixels, motion blur or JPEG blocks.

### Step 2 · Check, improve and play 🎛️
Everything you change lives in the 🎛️ TINKER ZONE of [`cleaning_denoise/action.yaml`](cleaning_denoise/action.yaml). Change one thing, push, and compare the job summary and `preview.png` with the run before.

- [ ] `method: "none"`: watch the noise gate fail. That's what the cheap camera really delivers.
- [ ] `median` versus `gaussian` at `kernel: "3"`: which one removes the dead pixels? Look at the *removed ×4* row.
- [ ] `kernel: "7"`: the noise drops further. Now look at the drop itself, and at the final test accuracy on Demo Day.
- [ ] Chain filters: `method: "[median, gaussian]"`. Try `bilateral` and `nlmeans`, and compare **ms per frame**. Would nlmeans run on a clip-on camera?
- [ ] 📓 Sweep kernel sizes and plot noise against sharpness in [`cleaning.ipynb`](cleaning.ipynb).

### Step 3 · Analyse, and decide if we pass on 🚦
- [ ] 🚦 **Gate:** estimated noise σ after cleaning ≤ 4.0.
- [ ] Compare *sharpness before* and *after*. Did you smooth away the drop together with the noise?
- [ ] A gate can pass while the product gets worse. Check Demo Day's test accuracy before you celebrate.

**Ready to pass on to Improving?** Gate green, sharpness still reasonable, accuracy not worse → pull request: *"Cleaning: what I changed and why"*.

⬅️ [Why this stage exists](README.md) · [The whole pipeline](../README.md)
