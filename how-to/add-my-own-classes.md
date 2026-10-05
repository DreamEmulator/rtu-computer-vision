# ❓ How to add my own classes?

> 🎯 **Goal** · the pipeline learns *your* classes from *your* footage, and its test accuracy is honest.<br>
> 🗺️ **Route** · 📷 1 Digital Data → 🧽 2 → 🔆 3 → ✂️ 4 → 🧬 5 → 🌳 6 → 🎤 7<br>
> 🧰 **You need** · a phone, about an hour, and PyCharm with the project's `.venv`

The pipeline doesn't know it's looking at drips. Folder names become class names, so `cow/`, `horse/` and `no_animal/` work just as well as `drop/`. This How To uses farm animals; swap in your own subject.

## 🪜 Steps

### 1 · Decide on the classes
- [ ] Two to four classes the camera can *see* the difference between. Folder names: lowercase, no spaces (`cow`, `horse`, `no_animal`).
- [ ] Add a "nothing" class. A model that only knows *cow* and *horse* will call an empty field one of the two.

### 2 · Film, don't photograph
- [ ] Per class, at least **three short videos** of 10–20 seconds, each in a different place, light or day. Walk around the subject while you film.
- [ ] Copy them to your laptop, e.g. `~/Desktop/clips/`. Videos stay out of the repository; only frames go in.

Why three videos? Stage 1 locks whole videos away as the test set. The model has to recognise a cow it has never seen, not the frame next to one it trained on.

### 3 · Turn videos into frames
In PyCharm's **Terminal** tab:

```bash
python -m tools.frames_from_video ~/Desktop/clips/cow_*.mp4   --label cow
python -m tools.frames_from_video ~/Desktop/clips/horse_*.mp4 --label horse
python -m tools.frames_from_video ~/Desktop/clips/field_*.mp4 --label no_animal
```

You get one frame every half second (`--every 0.5`), shrunk to 640 px (`--side 640`), in `data/raw/<label>/`. A 20-second video gives 40 frames named `cow-meadow_0000.jpg`, `cow-meadow_0001.jpg`, …: the part before the last `_` says which video a frame came from.

- [ ] Flip through `data/raw/cow/`. Blurry, empty, or a different animal? Delete it. Garbage in, garbage out.

### 4 · Point the pipeline at your frames
- [ ] In [`1_digital-data/action.yaml`](../1_digital-data/action.yaml):
  ```yaml
    source:
      default: "folder"
    group_by:
      default: "prefix"
  ```
- [ ] Swap the snap: replace `snaps/default_drip.jpg` with a photo of your own subject (not one of your video frames).
- [ ] ▶ **0 · Everything** in PyCharm's run menu (🚀 Stages).

### 5 · Follow the gates
Every stage was tuned for a drip chamber on a bright backlight. Your footage is different, and the first red gate tells you where. Fix the first red stage, run again, repeat. The [README](../README.md#the-stages) in that stage's folder tells you which values to try.

### 6 · Commit and let CI prove it
- [ ] `git add data/raw 1_digital-data snaps` → commit → push. CI reads the same frames from the repository.

## 🔀 If you see…

| You see | It means | Go to |
|---|---|---|
| ❌ gate `images per class` | fewer than 30 frames in a class | film another video, or `--every 0.25` |
| ❌ `cow comes from one video only` | `group_by: prefix` needs at least two videos per class: one for training, one for the test | film a second video |
| ❌ `empty mask ratio` or `full mask ratio` in Segmenting | the threshold expects a dark object on a bright wall | [stage 4](../4_segmenting/README.md): try `invert` and `threshold`, ▶ `segmenting.py` and look at the cow row of `before_after.png`; or `pass_on: "original"` |
| 99 % test accuracy with `group_by: file`, much less with `prefix` | the 99 % was a leak (see below) | trust the lower number |
| all gates green, but your snap gets the wrong answer | the test set doesn't look like the real world yet | [How to train our model on cows?](train-our-model-on-cows.md) |

## 🧠 Why `group_by: prefix`?

Two frames half a second apart are near-copies. With `group_by: "file"`, a random split puts frame 12 in training and frame 13 in the test: the model is graded on answers it has already seen. With `"prefix"`, every frame of one video lands on the same side, so the test set only holds videos the model has never seen.

Try it once: run with `file`, then with `prefix`, and compare the test accuracy. The difference is how much the first number lied.

## 🏆 Done when
- `source: folder`, `group_by: prefix`, and every gate green.
- You can say how many videos per class are in the test set: stage 1 reports `groups per class`.

## 💼 At work this is called…
- **Data leakage**: test data sneaking into training. In medical imaging, the same patient's scans on both sides of the split.
- **A grouped split**: "split by patient", "by video", "by day". In scikit-learn: `GroupKFold`, `StratifiedGroupKFold` (what stage 1 uses).
- **A data collection protocol**: deciding *how* to collect data so the test set looks like the real world.

⬅️ [All How To's](README.md) · [The whole pipeline](../README.md)
