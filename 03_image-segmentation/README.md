# CI Stage: Image Segmentation

<p align="center"><img src="pipeline.svg" alt="The Drip Detector CI-pipeline with Segmenting highlighted" width="760"></p>

## Why?

📍 **Where we are:** Processing, stage 4: **Segmenting**. Pre-Processing hands over here, from *make the image good* to *understand what's in it*.
On the layer diagram: Natural Scene → Digital Data → **Preparing for Analysis** → Model Training → Inference.

Until now every pixel counted the same. But the drip chamber is only part of the frame; the rest is wall, stand and tubing.

Segmenting decides which pixels belong to the object: the chamber, the fluid, the falling drop. That way the features of stage 5 describe the chamber, not the wallpaper, and a model can't cheat by learning the background.

**Without this stage:** the model learns whatever is easiest to see, and that's rarely the drop.

| | |
|---|---|
| 📅 Lecture | Tue 29.09 |
| 🛠 How? | [`segmenting_threshold/action.yaml`](segmenting_threshold/action.yaml) — the 🎛️ tinker zone |
| 📋 What? | [`Todo_Segmenting.md`](Todo_Segmenting.md) — Step 0 to Step 3 |
| 📓 Go deeper | [`segmenting.ipynb`](segmenting.ipynb) |

⬅️ [Back to the whole pipeline](../README.md)
