# CI Stage: Introduction to Image Processing

<p align="center"><img src="pipeline.svg" alt="The Drip Detector CI-pipeline with Digital Data highlighted" width="760"></p>

## Why?

📍 **Where we are:** Pre-Processing, stage 1: **Digital Data**, the first stop after the camera (Step 0).
On the layer diagram: Natural Scene → **Digital Data** → Preparing for Analysis → Model Training → Inference.

The investors' demo is a camera clipped to an IV drip chamber. But "the camera" is really many cameras: every hospital mounts it differently, some sideways, some in colour, all at different resolutions. A model can only learn from data that looks the same every time.

Digital Data turns whatever arrives, including the photo you snap in Step 0, into one small grid of numbers: 128 × 128 grey values. It also locks away a test set on day one that no model will ever train on, so every accuracy we report later is honest.

In 1957 Russell Kirsch scanned the first digital photograph: his baby son, 176 × 176 pixels. Our frames are smaller than that, and a runner in a data centre prepares hundreds of them per second.

**Without this stage:** every later stage has to cope with every camera, and nobody can trust the test score.

| | |
|---|---|
| 📅 Lecture | Tue 22.09 |
| 🛠 How? | [`digital-data_prepare/action.yaml`](digital-data_prepare/action.yaml) — the 🎛️ tinker zone |
| 📋 What? | [`Todo_Digital-Data.md`](Todo_Digital-Data.md) — Step 0 to Step 3 |
| 📓 Go deeper | [`digital-data.ipynb`](digital-data.ipynb) |

⬅️ [Back to the whole pipeline](../README.md)
