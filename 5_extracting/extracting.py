"""
🧬 Stage 5 · Extracting — how do we describe a frame with numbers a model can learn from?
════════════════════════════════════════════════════════════════════════════════════════
Lecture: Feature Extraction and Data Preparation · Thu 01.10 · Why: README.md · The values: action.yaml

This is the file CI runs for this stage. Press ▶ in PyCharm (or `python 5_extracting/extracting.py`):

    build/extracting/report.md          the values you passed in, every number explained, the gates
    build/extracting/before_after.png   your image for analyzing and one test frame per class: in → an augmented copy → HOG → LBP

Change one value in action.yaml, press ▶ again, and the report says what changed since your last run.

16 384 pixels in, a couple of thousand meaningful numbers out. Three ideas, in the order this file runs them:

    5d augment()      more training frames, made up from the ones we have (training frames only!)
    5a hog_features() · 5b lbp_features()   describe every frame as numbers
    5c fit_scaler()   put every number on the same ruler, measured on the training frames only
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # 🔒 finds stages/

import json
import time

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from stages.common import SNAP, to_luma
from stages.stage import Stage, picture, run   # 🔒 the plumbing: values, input, report, gates

KEY = "extracting"


def extract(gray: np.ndarray, stage_values: dict) -> dict[str, np.ndarray]:
    """{feature name: vector} for one single-channel frame. main() glues them into one vector."""
    out = {}
    for name in stage_values.get("features") or ["hog"]:
        if name == "hog":
            out["hog"] = hog_features(gray, stage_values)
        elif name == "lbp":
            out["lbp"] = lbp_features(gray, stage_values)
        elif name == "histogram":                                # how bright, ignoring where
            hist, _ = np.histogram(gray, bins=stage_values.get("histogram_bins", 32), range=(0, 256))
            out["histogram"] = hist / gray.size
        elif name == "pixels":                                   # the frame itself, shrunk to a thumbnail
            s = stage_values.get("pixels_size", 16)
            out["pixels"] = cv2.resize(gray, (s, s), interpolation=cv2.INTER_AREA).ravel() / 255.0
        else:
            raise SystemExit(f"❌ Unknown feature '{name}'. Choose hog | lbp | histogram | pixels")
    return out


# ── 5a · HOG: Histogram of Oriented Gradients ─────────────────────────────────────────────────────────
# 👾 Cut the frame into cells and ask every cell: which way do the edges point, and how strongly? Measure the
#    brightness slope at every pixel (Sobel), file it under one of `hog_orientations` directions weighted by
#    its steepness, and normalise blocks of cells so a dim and a bright frame of the same shape give the same
#    numbers (Dalal & Triggs, 2005, to find pedestrians). 0° = brightness changes left↔right: a vertical
#    edge, like the chamber's walls. 90° = it changes top↔bottom: a horizontal edge, like the fluid line.
#    Smaller cells see finer detail and make the vector explode: the 🚦 gate guards the camera chip's memory.

def hog_params(stage_values: dict) -> dict:
    p, c = stage_values.get("hog_pixels_per_cell", 16), stage_values.get("hog_cells_per_block", 2)
    return {"orientations": stage_values.get("hog_orientations", 9), "pixels_per_cell": (p, p), "cells_per_block": (c, c),
            "block_norm": "L2-Hys"}


def hog_features(gray: np.ndarray, stage_values: dict) -> np.ndarray:
    return hog(gray, feature_vector=True, **hog_params(stage_values))


# ── 5b · LBP: Local Binary Patterns ───────────────────────────────────────────────────────────────────
# 👾 Every pixel looks at P neighbours on a circle and writes a 1 for each that's at least as bright as
#    itself (Ojala, Pietikäinen & Harwood, Oulu, 1994–96). "Uniform" codes keep P + 2 kinds:
#    0 = bright spot, 1 … P−1 = edges and corners, P = flat or dark spot (a flat patch counts as "brighter",
#    so the masked black background lands here), P + 1 = noisy. Counting codes per cell of an lbp_grid ×
#    lbp_grid layout says what the texture is, and roughly where. HOG says which way; LBP says what it feels like.

def lbp_codes(gray: np.ndarray, stage_values: dict) -> np.ndarray:
    return local_binary_pattern(gray, stage_values.get("lbp_points", 8), stage_values.get("lbp_radius", 1), method="uniform")


def lbp_features(gray: np.ndarray, stage_values: dict) -> np.ndarray:
    p, grid = stage_values.get("lbp_points", 8), stage_values.get("lbp_grid", 4)
    codes = lbp_codes(gray, stage_values)
    bins, (h, w) = p + 2, gray.shape
    hists = []
    for gy in range(grid):
        for gx in range(grid):
            cell = codes[gy * h // grid:(gy + 1) * h // grid, gx * w // grid:(gx + 1) * w // grid]
            hist, _ = np.histogram(cell, bins=bins, range=(0, bins))
            hists.append(hist / max(cell.size, 1))
    return np.concatenate(hists)


# ── 5c · Scaling & normalization ──────────────────────────────────────────────────────────────────────
# 👾 Centimetres and kilometres. HOG numbers run from 0 to about 0.3, LBP shares from 0 to 1. A model that
#    measures distances between frames (kNN, an SVM) hears the biggest numbers shouting and the rest whispering.
#      minmax    (x − min) / (max − min): every feature squeezed into 0 … 1
#      standard  (x − μ) / σ, the z-score (Gauss): "how many standard deviations from an average frame"
#    The ruler is measured on the TRAINING frames only: measuring it on the test set would be peeking. So a test
#    frame brighter than any training frame lands above 1 with minmax. Correct, and a reason to prefer z-scores.
#    Trees don't care: they only ask "bigger than t?", and stretching a ruler moves t along with it.

def fit_scaler(X_train: np.ndarray, stage_values: dict):
    scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler()}.get(stage_values.get("scaling") or "none")
    return scaler.fit(X_train) if scaler is not None else None


# ── 5d · Augmentation: rotation, flipping, scaling ────────────────────────────────────────────────────
# 👾 We have 270 training frames and want more, without filming. So make plausible *different* camera frames
#    of the same situation: mirrored, tilted a few degrees, zoomed a little, a bit brighter or darker. Every
#    move is one 2 × 3 matrix (Euler's affine map: straight lines stay straight). The rule: only changes the
#    real camera could see. A drip chamber is never upside down, so augment_flip_vertical teaches the model
#    something that never happens. And only TRAINING frames get copies: the test set must look like the real
#    world, not like our tricks. Copies share a group with their original, so cross-validation in stage 6
#    never grades a copy of a frame it trained on.

def flip(img: np.ndarray, horizontal: bool, vertical: bool) -> np.ndarray:
    if horizontal:
        img = cv2.flip(img, 1)
    if vertical:
        img = cv2.flip(img, 0)
    return img


def affine(img: np.ndarray, degrees: float, zoom: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, zoom)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def augment(img: np.ndarray, stage_values: dict, rng: np.random.Generator) -> np.ndarray:
    """One random copy: maybe flipped, a little rotated and zoomed, a little brighter or darker."""
    img = flip(img, bool(stage_values.get("augment_flip_horizontal")) and rng.random() < 0.5,
               bool(stage_values.get("augment_flip_vertical")) and rng.random() < 0.5)
    lo, hi = stage_values.get("augment_scale") or [1.0, 1.0]
    r = stage_values.get("augment_rotate_degrees") or 0
    img = affine(img, rng.uniform(-r, r), rng.uniform(lo, hi))
    b = stage_values.get("augment_brightness") or 0
    if b:
        img = brightness(img, 1 + rng.uniform(-b, b))
    return img


def main() -> None:
    stage = Stage(KEY)                           # the values from action.yaml
    stage_values, src = stage.values, stage.frames()         # every frame stage 4 passed on, and your image for analyzing
    space, classes = src.color_space, src.classes
    rng = np.random.default_rng(stage.all_values["digital_data"].get("seed", 42))

    # 5d · Augment TRAIN frames only; the test set must look like the real world, not like our tricks.
    frames, labels, groups, splits = [], [], [], []
    augmented_preview = list(src.images)
    for i, (row, img) in enumerate(zip(src.rows, src.images)):
        frames.append(img); labels.append(row["label"]); groups.append(i); splits.append(row["split"])
        if row["split"] == "train":
            for k in range(int(stage_values.get("augment_copies") or 0)):
                aug = augment(img, stage_values, rng)
                frames.append(aug); labels.append(row["label"]); groups.append(i); splits.append("train")
                if k == 0:
                    augmented_preview[i] = aug
        elif stage_values.get("augment_copies"):
            augmented_preview[i] = augment(img, stage_values, rng)  # for the picture only, never used

    # 5a + 5b · Describe every frame
    t0 = time.perf_counter()
    vectors, dims = [], {}
    for f in frames:
        parts = extract(to_luma(f, space), stage_values)
        dims = {k: len(v) for k, v in parts.items()}
        vectors.append(np.concatenate(list(parts.values())))
    ms = 1000 * (time.perf_counter() - t0) / len(frames)
    X = np.asarray(vectors, dtype=np.float32)
    y = np.array([classes.index(l) if l in classes else -1 for l in labels])
    splits = np.array(splits)
    tr, te, sn = splits == "train", splits == "test", splits == SNAP

    # 5c · One ruler, measured on the training frames
    scaling = stage_values.get("scaling") or "none"
    X_train, X_test, X_snap = X[tr], X[te], X[sn]
    scaler = fit_scaler(X_train, stage_values)
    if scaler is not None:
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        X_snap = scaler.transform(X_snap) if len(X_snap) else X_snap

    # What stage 6 receives: the vectors for the forest, the pixels for the network
    I = np.stack(frames)
    np.savez_compressed(stage.dir / "features.npz", X_train=X_train, X_test=X_test, X_snap=X_snap,
                        y_train=y[tr], y_test=y[te], g_train=np.array(groups)[tr],
                        I_train=I[tr], I_test=I[te], I_snap=I[sn], classes=np.array(classes))
    (stage.dir / "meta.json").write_text(json.dumps({**src.meta, "feature_dims": dims}, indent=2))

    # 👀 before_after.png
    for i, img in enumerate(src.images):
        if (steps := stage.steps(i)) is not None:
            g = to_luma(img, space)
            picture(steps, "in: from stage 4", img, space)
            picture(steps, "5d augmented copy", augmented_preview[i], space)
            picture(steps, "5a HOG", hog_picture(g, stage_values))
            picture(steps, "5b LBP codes", lbp_picture(g, stage_values))

    nans = int(np.isnan(X).sum())
    gate = stage_values.get("gate_max_features")
    before, after = loud_share(X[tr]), loud_share(X_train)
    stage.metric("features", stage_values.get("features"))
    stage.metric("feature vector length", int(X.shape[1]), f"Numbers per frame. The 🚦 gate allows {gate} (the camera chip's memory).",
                 "good" if gate is None or X.shape[1] <= gate else "bad")
    stage.metric("dims per feature", dims)
    stage.metric("train frames (incl. augmented)", int(tr.sum()), "Training frames plus their augmented copies (augment_copies).")
    stage.metric("test frames", int(te.sum()), "Never augmented: if this number changes with augment_copies, something leaks.")
    stage.metric("scaling", scaling)
    stage.metric("loudest 10% of features, share of a distance", f"{before:.0f} % → {after:.0f} %",
                 "The widest tenth of the features: their share of the distance between two training frames, before → after "
                 "scaling. Lower = the other features get a say too (that's what kNN and SVM need).",
                 "good" if after < before - 5 or scaling == "none" and before < 20 else "check")
    stage.metric("NaN values", nans, "Numbers that aren't numbers. Must be 0.", "good" if nans == 0 else "bad")
    stage.perf("ms per frame", ms)
    stage.perf("pixels in → numbers out", f"{src.images[0].shape[0] * src.images[0].shape[1]} → {X.shape[1]}")

    if (i := src.snap_idx) is not None:
        stage.snap("feature vector length", int(X.shape[1]))
        g = to_luma(src.images[i], space)
        stage.snap_image("hog", hog_picture(g, stage_values), "gray")
        stage.snap_image("lbp", lbp_picture(g, stage_values), "gray")

    stage.tip("`features: \"[lbp]\"` only, then `\"[hog]\"` only. Which one carries the drop? Run `python run_pipeline.py --from extracting`.")
    stage.tip("`hog_pixels_per_cell: \"8\"`: four times the cells. Watch the vector length and the gate.")
    stage.tip("`scaling: \"none\"`: the forest won't care. Then try `model: \"knn\"` in 6_random-forest and watch it suffer.")
    stage.tip("`augment_flip_vertical: \"true\"`: drops falling upwards. Predict what happens to the accuracy, then check.")

    stage.gate("feature vector length", X.shape[1], max=gate)
    stage.gate("NaN values", nans, max=0)
    stage.finish(f"5 · Extracting — {' + '.join(stage_values.get('features') or [])}, {X.shape[1]} features")


# ── Helpers: pictures, names, a measurement ───────────────────────────────────────────────────────────

def hog_picture(gray: np.ndarray, stage_values: dict) -> np.ndarray:
    """HOG drawn as little stars: one line per direction, as bright as that direction is strong."""
    _, vis = hog(gray, visualize=True, **hog_params(stage_values))
    return np.clip(vis / max(vis.max(), 1e-9) * 255, 0, 255).astype(np.uint8)


def lbp_picture(gray: np.ndarray, stage_values: dict) -> np.ndarray:
    codes = lbp_codes(gray, stage_values)
    return (codes / max(codes.max(), 1) * 255).astype(np.uint8)


def loud_share(X: np.ndarray, share: float = 0.10, pairs: int = 500) -> float:
    """% of the squared distance between random pairs of frames that the widest `share` of the features cause."""
    if len(X) < 2:
        return 0.0
    loud = np.argsort(X.std(axis=0))[::-1][:max(1, int(share * X.shape[1]))]
    pick = np.random.default_rng(0).choice(len(X), size=(pairs, 2))
    d2 = (X[pick[:, 0]] - X[pick[:, 1]]) ** 2
    return float(100 * d2[:, loud].sum() / max(d2.sum(), 1e-12))


def feature_names(stage_values: dict, shape: tuple) -> list[tuple[str, tuple | None]]:
    """What every number in the vector describes → [(name, (y0, x0, y1, x1) region of the frame, or None)].
    Same order as extract(). Stage 6 uses it to say which features the forest relies on."""
    h, w = shape[:2]
    out = []
    for name in stage_values.get("features") or ["hog"]:
        if name == "hog":   # skimage's order: block row, block column, cell in block (row, column), direction
            p = hog_params(stage_values)
            ppc, cpb, o = p["pixels_per_cell"][0], p["cells_per_block"][0], p["orientations"]
            for by in range(h // ppc - cpb + 1):
                for bx in range(w // ppc - cpb + 1):
                    for cy in range(cpb):
                        for cx in range(cpb):
                            y, x = by + cy, bx + cx
                            out += [(f"HOG cell ({y},{x}) {180 * k // o}–{180 * (k + 1) // o}°",
                                     (y * ppc, x * ppc, (y + 1) * ppc, (x + 1) * ppc)) for k in range(o)]
        elif name == "lbp":
            pts, grid = stage_values.get("lbp_points", 8), stage_values.get("lbp_grid", 4)
            kinds = {0: "bright spots", pts: "flat or dark spots", pts + 1: "noisy texture"}
            for gy in range(grid):
                for gx in range(grid):
                    box = (gy * h // grid, gx * w // grid, (gy + 1) * h // grid, (gx + 1) * w // grid)
                    out += [(f"LBP cell ({gy},{gx}) {kinds.get(b, f'edges {b}/{pts}')}", box) for b in range(pts + 2)]
        elif name == "histogram":
            bins = stage_values.get("histogram_bins", 32)
            out += [(f"brightness {256 * b // bins}–{256 * (b + 1) // bins - 1}", None) for b in range(bins)]
        elif name == "pixels":
            s = stage_values.get("pixels_size", 16)
            out += [(f"pixel ({r},{c}) of the {s}×{s} thumbnail", (r * h // s, c * w // s, (r + 1) * h // s, (c + 1) * w // s))
                    for r in range(s) for c in range(s)]
    return out


if __name__ == "__main__":
    run(KEY, main)
