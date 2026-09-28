"""Stage 4 · ✂️ Segmenting — separate the drip chamber from the background.

Topic: Image Segmentation (Tue 29.09) — global / adaptive / Otsu thresholding, morphology, contours.
Knobs: 03_image-segmentation/segmenting_threshold/action.yaml
"""
from __future__ import annotations

import cv2
import numpy as np

from stages.common import (ImageSet, StageReport, contact_sheet, fresh_stage_dir, load_config, load_images, odd,
                           run_stage, save_images, to_luma)

KEY = "segmenting"
SHAPES = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}
MORPH = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE, "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}


def threshold(gray: np.ndarray, cfg: dict) -> tuple[np.ndarray, float]:
    """Returns (binary mask 0/255, threshold value used)."""
    method = cfg.get("threshold", "otsu")
    mode = cv2.THRESH_BINARY_INV if cfg.get("invert", True) else cv2.THRESH_BINARY
    if method == "global":
        t, mask = cv2.threshold(gray, cfg.get("global_value", 127), 255, mode)
    elif method == "otsu":
        t, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
    elif method == "triangle":
        t, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_TRIANGLE)
    elif method == "adaptive":
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode,
                                     odd(max(3, cfg.get("adaptive_block", 31))), cfg.get("adaptive_c", 5))
        t = float("nan")  # a different threshold for every neighbourhood
    else:
        raise SystemExit(f"❌ Unknown threshold '{method}'. Choose global | otsu | triangle | adaptive")
    return mask, float(t)


def morphology(mask: np.ndarray, cfg: dict) -> np.ndarray:
    k = odd(cfg.get("morph_kernel", 3))
    kernel = cv2.getStructuringElement(SHAPES[cfg.get("morph_shape", "ellipse")], (k, k))
    for op in cfg.get("morphology") or []:
        mask = cv2.morphologyEx(mask, MORPH[op], kernel)
    return mask


def keep_contours(mask: np.ndarray, cfg: dict) -> tuple[np.ndarray, list]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [k for k in contours if cv2.contourArea(k) >= cfg.get("min_contour_area", 8)]
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, kept, -1, 255, thickness=cv2.FILLED)
    return (filled if cfg.get("fill_holes") else cv2.bitwise_and(mask, filled)), kept


def segment(img: np.ndarray, cfg: dict, color_space: str = "gray") -> tuple[np.ndarray, list, float]:
    """The whole stage for one frame → (mask, contours, threshold)."""
    mask, t = threshold(to_luma(img, color_space), cfg)
    mask, contours = keep_contours(morphology(mask, cfg), cfg)
    return mask, contours, t


def apply(img: np.ndarray, mask: np.ndarray, contours: list, how: str) -> np.ndarray:
    """What the next stage receives."""
    if how == "original":
        return img
    if how == "mask":
        return mask
    if how == "masked":
        return cv2.bitwise_and(img, img, mask=mask)
    if how == "crop":  # zoom in on the biggest object — hopefully the drip chamber
        if not contours:
            return img
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return cv2.resize(img[y:y + h, x:x + w], img.shape[1::-1], interpolation=cv2.INTER_AREA)
    raise SystemExit(f"❌ Unknown pass_on '{how}'. Choose masked | mask | crop | original")


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
        mask, contours, t = segment(img, c, space)
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
