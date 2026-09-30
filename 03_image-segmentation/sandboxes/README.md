# 🧪 Sandboxes · Image Segmentation

One image in, one image out. Each sandbox is a short Python script that does **one** step of stage 4 · Segmenting on **one** photo, so you can try an idea in a second instead of waiting for CI.

```
stage_3_improving.png ──▶ 4a threshold ──▶ 4b morphology ──▶ 4c contours ──▶ 4d boundary ──▶ stage 5
 (or snaps/default_drip.jpg)  mask            repaired mask     filled objects    crust only
```

| | Sandbox | 👾 The Robot… | 📥 Reads by default | 📤 Writes |
|---|---|---|---|---|
| 4a | [`sandbox_4a_threshold.py`](sandbox_4a_threshold.py) | sorts pixels into a dark and a light bin | `stage_3_improving.png` (else the default snap) | `stage_4a_threshold.png` |
| 4b | [`sandbox_4b_morphology.py`](sandbox_4b_morphology.py) | planes and fills the mask like a woodworker | `stage_4a_threshold.png` | `stage_4b_morphology.png` · `stage_4b_changes.png` 👀 |
| 4c | [`sandbox_4c_contours.py`](sandbox_4c_contours.py) | traces every blob like your hand on paper | `stage_4b_morphology.png` | `stage_4c_contours.png` · `stage_4c_overlay.png` 👀 |
| 4d | [`sandbox_4d_boundary.py`](sandbox_4d_boundary.py) | scoops out the cookie, keeps the crust | `stage_4c_contours.png` | `stage_4d_boundary.png` · `stage_4d_overlay.png` 👀 |

Everything lands in `build/sandbox/`. That folder is in `.gitignore`, so your experiments stay on your machine.

## How?

From the repository root, after `pip install -r requirements.txt`:

```bash
python 03_image-segmentation/sandboxes/sandbox_4a_threshold.py              # the default snap, or stage 3's output
python 03_image-segmentation/sandboxes/sandbox_4a_threshold.py my_photo.jpg # your own photo
python 03_image-segmentation/sandboxes/sandbox_4b_morphology.py             # picks up 4a's output by itself
python 03_image-segmentation/sandboxes/sandbox_4c_contours.py
python 03_image-segmentation/sandboxes/sandbox_4d_boundary.py
```

In PyCharm: open a sandbox and press ▶. Each script only needs OpenCV and NumPy, so it also runs in a Colab cell.

## What?

1. Change **one** value in the 🎛️ TINKER ZONE at the top of a sandbox.
2. Run it (and the sandboxes after it, if you want to see the knock-on effect).
3. Open the new `build/sandbox/<unix time>_results_stage_4x.md`. It says what the Robot did, which knobs it used, every metric with 🟢 🟠 🔴, whether high or low is good and why, and what to try next. The unix time sorts your runs oldest to newest, so compare two files side by side.
4. Found something that works? Carry the idea over to the pipeline's tinker zone in [`../segmenting_threshold/action.yaml`](../segmenting_threshold/action.yaml) and let CI prove it (see [`../Todo_Segmenting.md`](../Todo_Segmenting.md)).

⚠️ The sandboxes work on one photo at up to 800 px. The pipeline works on hundreds of 128 × 128 frames. Ideas carry over; numbers don't always. Areas grow with the square of the image size: the pipeline's `min_contour_area: 8` on a 128-px frame is about 300 px² on an 800-px photo.

⬅️ [Why this stage exists](../README.md) · [The whole pipeline](../../README.md)
