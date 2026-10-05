# Your own frames go here

Set `source: "folder"` in [`1_digital-data/action.yaml`](../../1_digital-data/action.yaml) and put one folder per class:

```
data/raw/
├── drop/        frame_001.jpg, frame_002.jpg, …
├── no_drop/     …
└── low_fluid/   …
```

Folder names become the class names, so any subject works (`ripe/`, `unripe/`, …).

Filming is quicker than photographing: `python -m tools.frames_from_video clip.mp4 --label cow` turns a video into frames, and `group_by: "prefix"` keeps every video on one side of the train/test split. The whole route: [How to add my own classes?](../../how-to/add-my-own-classes.md)
Stage 1 accepts jpg, png, bmp, tif and webp, in any resolution.
Aim for at least 30 frames per class. For more than a few hundred MB, use
[Git LFS](https://git-lfs.com) or download the data in stage 1 instead of committing it.
