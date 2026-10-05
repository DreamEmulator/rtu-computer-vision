"""Stage 5 · 🧬 Extracting — 🔒 plumbing: augment the training frames, describe every frame, scale, previews, gates.

Topic: Feature Extraction and Data Preparation (Thu 01.10) — HOG, LBP, scaling, augmentation.
The maths: 04_feature-extraction/extracting_describe/recipe.py (yours to change)
The knobs: 04_feature-extraction/extracting_describe/action.yaml
"""
from __future__ import annotations

import json
import time

import numpy as np
from skimage.feature import hog

from stages.common import SNAP, StageReport, contact_sheet, fresh_stage_dir, load_config, load_images, load_recipe, run_stage, to_luma

KEY = "extracting"
recipe = load_recipe(KEY)
augment, extract, hog_params, lbp_features = (  # the notebook and older code import these from here
    recipe.augment, recipe.extract, recipe.hog_params, recipe.lbp_features)


def feature_names(cfg: dict, shape: tuple) -> list[tuple[str, tuple | None]]:
    """What every number in the feature vector describes → [(name, (y0, x0, y1, x1) region of the frame, or None)].

    Same order as extract(), so feature i of the vector is entry i of this list.
    """
    h, w = shape[:2]
    out = []
    for name in cfg.get("features") or ["hog"]:
        if name == "hog":   # skimage's order: block row, block column, cell in block (row, column), direction
            p = recipe.hog_params(cfg)
            ppc, cpb, o = p["pixels_per_cell"][0], p["cells_per_block"][0], p["orientations"]
            for by in range(h // ppc - cpb + 1):
                for bx in range(w // ppc - cpb + 1):
                    for cy in range(cpb):
                        for cx in range(cpb):
                            y, x = by + cy, bx + cx
                            out += [(f"HOG cell ({y},{x}) {180 * k // o}–{180 * (k + 1) // o}°",
                                     (y * ppc, x * ppc, (y + 1) * ppc, (x + 1) * ppc)) for k in range(o)]
        elif name == "lbp":
            pts, grid = cfg.get("lbp_points", 8), cfg.get("lbp_grid", 4)
            kinds = {0: "bright spots", pts: "flat or dark spots", pts + 1: "noisy texture"}
            for gy in range(grid):
                for gx in range(grid):
                    box = (gy * h // grid, gx * w // grid, (gy + 1) * h // grid, (gx + 1) * w // grid)
                    out += [(f"LBP cell ({gy},{gx}) {kinds.get(b, f'edges {b}/{pts}')}", box) for b in range(pts + 2)]
        elif name == "histogram":
            bins = cfg.get("histogram_bins", 32)
            out += [(f"brightness {256 * b // bins}–{256 * (b + 1) // bins - 1}", None) for b in range(bins)]
        elif name == "pixels":
            s = cfg.get("pixels_size", 16)
            out += [(f"pixel ({r},{c}) of the {s}×{s} thumbnail", (r * h // s, c * w // s, (r + 1) * h // s, (c + 1) * w // s))
                    for r in range(s) for c in range(s)]
    return out


def hog_picture(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """HOG drawn as little stars: one line per direction, as bright as that direction is strong."""
    _, vis = hog(gray, visualize=True, **recipe.hog_params(cfg))
    return np.clip(vis / max(vis.max(), 1e-9) * 255, 0, 255).astype(np.uint8)


def main() -> None:
    cfg = load_config()
    c = cfg[KEY]
    report = StageReport(KEY, cfg)
    src = load_images("segmenting")
    out_dir = fresh_stage_dir(KEY)
    space, classes = src.color_space, src.classes
    rng = np.random.default_rng(cfg["digital_data"].get("seed", 42))

    # Augment TRAIN frames only; the test set must look like the real world, not like our tricks.
    frames, labels, groups, splits = [], [], [], []
    augmented_preview = list(src.images)
    for i, (row, img) in enumerate(zip(src.rows, src.images)):
        frames.append(img); labels.append(row["label"]); groups.append(i); splits.append(row["split"])
        if row["split"] == "train":
            for k in range(int(c.get("augment_copies") or 0)):
                aug = recipe.augment(img, c, rng)
                frames.append(aug); labels.append(row["label"]); groups.append(i); splits.append("train")
                if k == 0:
                    augmented_preview[i] = aug
        elif c.get("augment_copies"):
            augmented_preview[i] = recipe.augment(img, c, rng)  # preview only, never used

    t0 = time.perf_counter()
    vectors, dims = [], {}
    for f in frames:
        parts = recipe.extract(to_luma(f, space), c)
        dims = {k: len(v) for k, v in parts.items()}
        vectors.append(np.concatenate(list(parts.values())))
    ms = 1000 * (time.perf_counter() - t0) / len(frames)
    X = np.asarray(vectors, dtype=np.float32)
    y = np.array([classes.index(l) if l in classes else -1 for l in labels])
    splits = np.array(splits)
    tr, te, sn = splits == "train", splits == "test", splits == SNAP

    scaling = c.get("scaling") or "none"
    X_train, X_test, X_snap = X[tr], X[te], X[sn]
    scaler = recipe.fit_scaler(X_train, c)                 # fitted on train only — no peeking at the test set
    if scaler is not None:
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        X_snap = scaler.transform(X_snap) if len(X_snap) else X_snap

    I = np.stack(frames)
    np.savez_compressed(out_dir / "features.npz", X_train=X_train, X_test=X_test, X_snap=X_snap,
                        y_train=y[tr], y_test=y[te], g_train=np.array(groups)[tr],
                        I_train=I[tr], I_test=I[te], I_snap=I[sn], classes=np.array(classes))
    (out_dir / "meta.json").write_text(json.dumps({**src.meta, "feature_dims": dims}, indent=2))

    hog_vis, lbp_vis = [], []
    for img in src.images:
        g = to_luma(img, space)
        hog_vis.append(hog_picture(g, c))
        codes = recipe.lbp_codes(g, c)
        lbp_vis.append((codes / max(codes.max(), 1) * 255).astype(np.uint8))
    contact_sheet(out_dir / "preview.png", src.rows,
                  [("input", src.images, space), ("augmented", augmented_preview, space),
                   ("HOG", hog_vis, "gray"), ("LBP codes", lbp_vis, "gray")],
                  f"5 · Extracting — {' + '.join(c.get('features') or [])}, {X.shape[1]} features")

    nans = int(np.isnan(X).sum())
    report.metric("features", c.get("features"))
    report.metric("feature vector length", int(X.shape[1]))
    report.metric("dims per feature", dims)
    report.metric("train frames (incl. augmented)", int(tr.sum()))
    report.metric("test frames", int(te.sum()))
    report.metric("scaling", scaling)
    report.metric("NaN values", nans)
    report.perf("ms per frame", ms)
    report.perf("pixels in → numbers out", f"{src.images[0].shape[0] * src.images[0].shape[1]} → {X.shape[1]}")

    if (i := src.snap_idx) is not None:
        report.snap("feature vector length", int(X.shape[1]))
        report.snap_image("hog", hog_vis[i], "gray")
        report.snap_image("lbp", lbp_vis[i], "gray")

    report.gate("feature vector length", X.shape[1], max=c.get("gate_max_features"))
    report.gate("NaN values", nans, max=0)
    report.finish()


if __name__ == "__main__":
    run_stage(KEY, main)
