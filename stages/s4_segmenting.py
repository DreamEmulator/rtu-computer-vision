"""Stage 4 · ✂️ Segmenting — 🔒 plumbing: every frame through the recipe, then previews, metrics and gates.

Topic: Image Segmentation (Tue 29.09) — global / adaptive / Otsu thresholding, morphology, contours.
The maths: 03_image-segmentation/segmenting_threshold/recipe.py (yours to change)
The knobs: 03_image-segmentation/segmenting_threshold/action.yaml
"""
from __future__ import annotations

import cv2
import numpy as np

from stages.common import (ImageSet, StageReport, contact_sheet, fresh_stage_dir, load_config, load_images,
                           load_recipe, run_stage, save_images, to_luma)

KEY = "segmenting"
recipe = load_recipe(KEY)
threshold, morphology, keep_contours, boundary, apply = (  # the notebook imports these from here
    recipe.threshold, recipe.morphology, recipe.keep_contours, recipe.boundary, recipe.apply)


def overlay(img: np.ndarray, mask: np.ndarray, contours: list, color_space: str) -> np.ndarray:
    vis = cv2.cvtColor(to_luma(img, color_space), cv2.COLOR_GRAY2BGR)
    tint = vis.copy()
    tint[mask > 0] = (60, 200, 60)
    vis = cv2.addWeighted(vis, 0.6, tint, 0.4, 0)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 1)
    return vis


def main() -> None:
    import time
    cfg = load_config()
    c = cfg[KEY]
    report = StageReport(KEY, cfg)
    src = load_images("improving")
    out_dir = fresh_stage_dir(KEY)
    space, how = src.color_space, c.get("pass_on", "masked")

    masks, outputs, overlays, n_contours, thresholds = [], [], [], [], []
    t0 = time.perf_counter()
    for img in src.images:
        mask, contours, t = recipe.segment(to_luma(img, space), c)
        masks.append(mask); outputs.append(apply(img, mask, contours, how))
        overlays.append(overlay(img, mask, contours, space)); n_contours.append(len(contours)); thresholds.append(t)
    ms = 1000 * (time.perf_counter() - t0) / len(src.images)

    meta = {**src.meta, "color_space": "gray" if how == "mask" else space}
    save_images(KEY, ImageSet(src.rows, outputs, meta))
    contact_sheet(out_dir / "preview.png", src.rows,
                  [("input", src.images, space), ("mask + contours", overlays, "bgr"), (how, outputs, meta["color_space"])],
                  f"4 · Segmenting — {c.get('threshold')} threshold, morphology {c.get('morphology')}")

    fg = np.array([(masks[i] > 0).mean() for i in src.data_idx])
    report.metric("threshold method", c.get("threshold"))
    data_t = [thresholds[i] for i in src.data_idx]
    if not np.all(np.isnan(data_t)):
        report.metric("mean threshold value", float(np.nanmean(data_t)))
    report.metric("foreground %", 100 * fg.mean())
    report.metric("contours per frame", float(np.mean([n_contours[i] for i in src.data_idx])))
    report.metric("empty masks %", 100 * (fg < 0.005).mean())
    report.metric("full masks %", 100 * (fg > 0.95).mean())
    if (c.get("boundary") or "none") != "none":
        report.metric("boundary", c.get("boundary"))
    report.metric("passed on", how)
    report.perf("ms per frame", ms)

    if (i := src.snap_idx) is not None:
        report.snap("foreground %", 100 * (masks[i] > 0).mean())
        report.snap("contours", n_contours[i])
        report.snap_image("overlay", overlays[i], "bgr")
        report.snap_image("output", outputs[i], meta["color_space"])

    report.gate("empty mask ratio", (fg < 0.005).mean(), max=c.get("gate_max_empty_ratio"))
    report.gate("full mask ratio", (fg > 0.95).mean(), max=c.get("gate_max_full_ratio"))
    report.finish()


if __name__ == "__main__":
    run_stage(KEY, main)
