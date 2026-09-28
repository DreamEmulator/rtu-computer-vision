"""Helpers for the notebooks (CI doesn't use this file)."""
from __future__ import annotations

import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np

from stages.common import load_config, to_display


def prepare(upto: str) -> dict:
    """Run the stages up to and including `upto` with your current action.yaml knobs — what CI would see."""
    from run_pipeline import run
    with contextlib.redirect_stdout(io.StringIO()):
        ok = run(None, upto, keep_going=True)
    print(f"✔ Stages up to '{upto}' ran with your tinker-zone values"
          + ("" if ok else "  (⚠ a quality gate failed — see build/*/metrics.json)"))
    return load_config()


def run_stage(key: str) -> None:
    """Run one stage quietly (its inputs must already exist)."""
    from run_pipeline import run
    with contextlib.redirect_stdout(io.StringIO()):
        run(key, key, keep_going=True)


def show(images, titles=None, cols=None, color_space: str = "gray", size: float = 2.6, suptitle=None):
    """Plot a row (or grid) of images; 2-D arrays are shown in gray."""
    n = len(images)
    cols = cols or n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows + (0.4 if suptitle else 0)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, img in enumerate(images):
        ax = axes.ravel()[k]
        im = to_display(img, color_space) if img.ndim == 3 else img
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None, **({"vmin": 0, "vmax": 255} if im.dtype == np.uint8 else {}))
        if titles:
            ax.set_title(titles[k], fontsize=9)
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()
