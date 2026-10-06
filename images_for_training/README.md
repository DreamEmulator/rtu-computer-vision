# 🏋️ Images for training

Your own classes go here: one folder per class, the images inside. Then set `source: "folder"` in [`1_digital-data/action.yaml`](../1_digital-data/action.yaml).

```
images_for_training/
├── drop/        frame_001.jpg, frame_002.jpg, …
├── no_drop/     …
└── low_fluid/   …
```

- **Folder names become the class names**, so any subject works (`cow/`, `horse/`, `ripe/`, …).
- **Stage 1 locks away part of these as the test set** (`test_size`): images the model never learns from, so its test accuracy is honest. Everything else is what it learns from.
- **Until you add your own,** stage 1 draws a synthetic day-zero set of drip chambers itself (`source: "synthetic"`, in `build/synthetic/`).
- **Film, don't photograph:** `python -m tools.frames_from_video clip.mp4 --label cow` turns a video into frames here, and `group_by: "prefix"` keeps every video on one side of the train/test split. The whole route: [How to add my own classes?](../how-to/add-my-own-classes.md)
- Stage 1 accepts jpg, png, bmp, tif and webp, in any resolution. Aim for at least 30 images per class. For more than a few hundred MB, use [Git LFS](https://git-lfs.com).

The images you want the pipeline to *analyze* (Step 0) go in [`images_for_analyzing/`](../images_for_analyzing/).
