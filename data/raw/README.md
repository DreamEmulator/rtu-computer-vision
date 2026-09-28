# Your own frames go here

Set `source: "folder"` in the tinker zone of [`01_image-processing/digital-data_prepare/action.yaml`](../../01_image-processing/digital-data_prepare/action.yaml) and put one folder per class:

```
data/raw/
├── drop/        frame_001.jpg, frame_002.jpg, …
├── no_drop/     …
└── low_fluid/   …
```

Folder names become the class names, so any subject works (`ripe/`, `unripe/`, …).
Stage 1 accepts jpg, png, bmp, tif and webp, in any resolution.
Aim for at least 30 frames per class. For more than a few hundred MB, use
[Git LFS](https://git-lfs.com) or download the data in stage 1 instead of committing it.
