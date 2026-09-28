"""Stage 3 · 🔆 Improving — make the important structure easy to see.

Topic: Image Preprocessing Methods (Thu 24.09) — histogram equalization, contrast, gradients and Canny.
Knobs: 02_image-preprocessing/improving_enhance/action.yaml
"""
from __future__ import annotations

import cv2
import numpy as np

from stages.common import (ImageSet, StageReport, contact_sheet, fresh_stage_dir, load_config, load_images, on_luma,
                           run_stage, save_images, timed, to_luma)

KEY = "improving"


def enhance(img: np.ndarray, cfg: dict, color_space: str = "gray") -> np.ndarray:
    method = cfg.get("method", "clahe")

    def fn(ch: np.ndarray) -> np.ndarray:
        if method in (None, "none"):
            return ch
        if method == "equalize":
            return cv2.equalizeHist(ch)
        if method == "clahe":
            t = int(cfg.get("clahe_tile", 8))
            return cv2.createCLAHE(clipLimit=float(cfg.get("clahe_clip", 2.0)), tileGridSize=(t, t)).apply(ch)
        if method == "stretch":
            lo, hi = np.percentile(ch, cfg.get("stretch_percentiles") or [1, 99])
            return np.clip((ch.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)
        if method == "gamma":
            lut = (255.0 * (np.arange(256) / 255.0) ** float(cfg.get("gamma", 1.0))).astype(np.uint8)
            return cv2.LUT(ch, lut)
        raise SystemExit(f"❌ Unknown method '{method}'. Choose none | equalize | clahe | stretch | gamma")

    return on_luma(img, color_space, fn)


def edges(img: np.ndarray, cfg: dict, color_space: str = "gray") -> np.ndarray:
    return cv2.Canny(to_luma(img, color_space), cfg.get("canny_low", 50), cfg.get("canny_high", 150))


def contrast(img: np.ndarray, color_space: str = "gray") -> float:
    return float(to_luma(img, color_space).std())


def main() -> None:
    cfg = load_config()
    c = cfg[KEY]
    report = StageReport(KEY, cfg)
    src = load_images("cleaning")
    out_dir = fresh_stage_dir(KEY)
    space = src.color_space

    enhanced, ms = timed(lambda im: enhance(im, c, space), src.images)
    edge_maps, ms_edges = timed(lambda im: edges(im, c, space), enhanced)
    pass_on = c.get("pass_on", "enhanced")
    if pass_on == "edges":
        save_images(KEY, ImageSet(src.rows, edge_maps, {**src.meta, "color_space": "gray"}))
    else:
        save_images(KEY, ImageSet(src.rows, enhanced, src.meta))
    contact_sheet(out_dir / "preview.png", src.rows,
                  [("input", src.images, space), (str(c.get("method")), enhanced, space), ("canny", edge_maps, "gray")],
                  f"3 · Improving — {c.get('method')} → passing on: {pass_on}")

    before = np.array([contrast(im, space) for im in src.data()])
    after = np.array([contrast(im, space) for im in src.data(enhanced)])
    report.metric("method", c.get("method"))
    report.metric("contrast before (σ, mean)", before.mean())
    report.metric("contrast after (σ, mean)", after.mean())
    report.metric("contrast of dimmest 5% before", np.percentile(before, 5))
    dim_after = report.metric("contrast of dimmest 5% after", np.percentile(after, 5))
    report.metric("edge pixels %", 100 * np.mean([(e > 0).mean() for e in src.data(edge_maps)]))
    report.metric("passed on", pass_on)
    report.perf("ms per frame (enhance)", ms)
    report.perf("ms per frame (Canny)", ms_edges)
    report.note("HOG normalises contrast per block too, so better-looking frames don't always mean better accuracy.")

    if (i := src.snap_idx) is not None:
        report.snap("contrast before", contrast(src.images[i], space))
        report.snap("contrast after", contrast(enhanced[i], space))
        report.snap("edge pixels %", 100 * (edge_maps[i] > 0).mean())
        report.snap_image("output", enhanced[i], space)
        report.snap_image("edges", edge_maps[i], "gray")

    report.gate("contrast of dimmest 5% frames", dim_after, min=c.get("gate_min_dim_contrast"))
    report.finish()


if __name__ == "__main__":
    run_stage(KEY, main)
