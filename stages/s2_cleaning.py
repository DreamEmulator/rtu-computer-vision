"""Stage 2 · 🧽 Cleaning — remove what the cheap camera added.

Topic: Image Preprocessing Methods (Thu 24.09) — Gaussian and median filters.
Knobs: 02_image-preprocessing/cleaning_denoise/action.yaml
"""
from __future__ import annotations

import cv2
import numpy as np

from stages.common import (ImageSet, StageReport, contact_sheet, fresh_stage_dir, load_config, load_images, odd,
                           run_stage, save_images, timed)

KEY = "cleaning"
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)  # difference of two Laplacians


def estimate_noise(img: np.ndarray) -> float:
    """Immerkær (1996): a fast estimate of Gaussian noise σ that ignores smooth image structure."""
    if img.ndim == 3:
        return float(np.mean([estimate_noise(img[..., k]) for k in range(img.shape[2])]))
    h, w = img.shape
    response = cv2.filter2D(img.astype(np.float64), -1, _NOISE_KERNEL)[1:-1, 1:-1]
    return float(np.sum(np.abs(response)) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2)))


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian — high for crisp edges (and, sadly, for noise)."""
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def clean(img: np.ndarray, cfg: dict) -> np.ndarray:
    methods = cfg.get("method", "median")
    for method in methods if isinstance(methods, list) else [methods]:
        k = odd(cfg.get("kernel", 3))
        if method in (None, "none"):
            continue
        elif method == "gaussian":
            img = cv2.GaussianBlur(img, (k, k), cfg.get("sigma") or 0)
        elif method == "median":
            img = cv2.medianBlur(img, max(3, k))
        elif method == "bilateral":
            img = cv2.bilateralFilter(img, cfg.get("bilateral_diameter", 7),
                                      cfg.get("bilateral_sigma_color", 50), cfg.get("bilateral_sigma_space", 50))
        elif method == "nlmeans":
            img = cv2.fastNlMeansDenoising(img, None, cfg.get("nlmeans_h", 10), 7, 21)
        else:
            raise SystemExit(f"❌ Unknown method '{method}'. Choose none | gaussian | median | bilateral | nlmeans")
    return img


def main() -> None:
    cfg = load_config()
    c = cfg[KEY]
    report = StageReport(KEY, cfg)
    src = load_images("digital_data")
    out_dir = fresh_stage_dir(KEY)
    space = src.color_space

    cleaned, ms = timed(lambda im: clean(im, c), src.images)
    save_images(KEY, ImageSet(src.rows, cleaned, src.meta))
    removed = [np.clip(cv2.absdiff(a, b).astype(np.int32) * 4, 0, 255).astype(np.uint8) for a, b in zip(src.images, cleaned)]
    contact_sheet(out_dir / "preview.png", src.rows,
                  [("input", src.images, space), ("cleaned", cleaned, space), ("removed ×4", removed, space)],
                  f"2 · Cleaning — {c.get('method')} (kernel {odd(c.get('kernel', 3))})")

    noise_after = np.mean([estimate_noise(im) for im in src.data(cleaned)])
    report.metric("method", c.get("method"))
    report.metric("noise σ before", np.mean([estimate_noise(im) for im in src.data()]))
    report.metric("noise σ after", noise_after)
    report.metric("sharpness before", np.mean([sharpness(im) for im in src.data()]))
    report.metric("sharpness after", np.mean([sharpness(im) for im in src.data(cleaned)]))
    report.perf("ms per frame", ms)
    report.note("Denoising always trades noise for detail: watch both numbers, not just one.")

    if (i := src.snap_idx) is not None:
        report.snap("noise σ before", estimate_noise(src.images[i]))
        report.snap("noise σ after", estimate_noise(cleaned[i]))
        report.snap_image("output", cleaned[i], space)
        report.snap_image("removed", removed[i], space)

    report.gate("noise σ after cleaning", noise_after, max=c.get("gate_max_noise"))
    report.finish()


if __name__ == "__main__":
    run_stage(KEY, main)
