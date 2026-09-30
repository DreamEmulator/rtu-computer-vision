# Todo · Demo Day

## What?

> **Step 1:** Collect underpants. **Step 2:** ? **Step 3:** Profit.
> — the Underpants Gnomes' business plan. Ours has a Step 0, and we actually know what Step 2 is.

### Step 0 · Snap 📸
- [ ] Open **[sdux.tech/computer-vision](https://sdux.tech/computer-vision?repo=DreamEmulator/rtu-computer-vision)**, snap a picture and upload it into the pipeline.
- [ ] Watch the webhooks fire. For **Demo Day** the page shows: `pipeline.finished`: the overall verdict, every stage's status and time, and both models' predictions for your snap. The whole journey, camera to verdict.

<sub>No phone at hand? Run `python run_pipeline.py --snap photo.jpg` locally, or open **Actions → 🚀 CI-Pipeline → Run workflow** and paste an image URL into `snap_url`.</sub>

### Step 1 · Input from the previous stage 📥
Every stage's artifact. Demo Day runs even when a gate failed, so the report always shows where the pipeline stopped.

### Step 2 · Check, improve and play 🎛️
Everything you change lives in the 🎛️ TINKER ZONE of [`demo-day_report/action.yaml`](demo-day_report/action.yaml). Change one thing, push, and compare the job summary and `preview.png` with the run before.

- [ ] Name your startup and write a pitch an investor remembers: `project_name`, `pitch`.
- [ ] Snap three photos, one of each class. How many does the pipeline get right? Real photos against synthetic training data: that gap has a name, *domain shift*.
- [ ] Where does the time go? Compare the durations of every stage on the page.
- [ ] Put the report online: **Settings → Pages → Source: GitHub Actions**, then add the repository variable `ENABLE_PAGES` = `true`.

### Step 3 · Analyse, and decide if we pass on 🚦
- [ ] 🚦 Every gate green? The report's verdict says **"Every gate passed. Ship it."**
- [ ] For every stage, one sentence each: why it exists, how you tuned it, what changed.

**Ready for the investors (Tue 13.10) and the exam (Thu 15.10)?** Three minutes: the pipeline, the one change per stage that mattered most, and the thing that surprised you.

⬅️ [Why this stage exists](README.md) · [The whole pipeline](../README.md)
