"""
🧬 Stage 5 · Extracting — the recipe
════════════════════════════════════
The maths of this stage and nothing else: one frame in, one vector of numbers out. CI runs exactly this file:

    .github/workflows/pipeline.yml
      └─ extracting_describe/action.yaml    🎛️ knobs + 🚦 gates
           └─ stages/s5_extracting.py        🔒 plumbing: every frame, the train/test rule, previews, gates
                └─ recipe.py                 🧪 you are here

The sandboxes next to this file call these same functions on real frames, one idea at a time:

    5a hog_features() · 5b lbp_features() → 5c fit_scaler() · 5d augment()

In the pipeline the order is: augment the training frames (5d) → extract (5a, 5b) → scale (5c).

Every function gets `knobs`: the TINKER ZONE of action.yaml (plus a sandbox's TRY).
Change a knob there; change the *maths* here. Add a feature of your own, say "edges" (the share of
Canny edge pixels per cell), and let the classifier tell you whether it was worth its numbers.

▶ Press ▶ on this file: your recipe runs on every frame, then the random forest takes the exam.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def extract(gray: np.ndarray, knobs: dict) -> dict[str, np.ndarray]:
    """{feature name: vector} for one single-channel frame. The pipeline glues them into one vector."""
    out = {}
    for name in knobs.get("features") or ["hog"]:
        if name == "hog":
            out["hog"] = hog_features(gray, knobs)
        elif name == "lbp":
            out["lbp"] = lbp_features(gray, knobs)
        elif name == "histogram":                                # how bright, ignoring where
            hist, _ = np.histogram(gray, bins=knobs.get("histogram_bins", 32), range=(0, 256))
            out["histogram"] = hist / gray.size
        elif name == "pixels":                                   # the frame itself, shrunk to a thumbnail
            s = knobs.get("pixels_size", 16)
            out["pixels"] = cv2.resize(gray, (s, s), interpolation=cv2.INTER_AREA).ravel() / 255.0
        else:
            raise SystemExit(f"❌ Unknown feature '{name}'. Choose hog | lbp | histogram | pixels")
    return out


# ── 5a · HOG ───────────────── 👾 in every cell: which way do the edges point, and how strongly?

def hog_params(knobs: dict) -> dict:
    p, c = knobs.get("hog_pixels_per_cell", 16), knobs.get("hog_cells_per_block", 2)
    return {"orientations": knobs.get("hog_orientations", 9), "pixels_per_cell": (p, p), "cells_per_block": (c, c),
            "block_norm": "L2-Hys"}


def hog_features(gray: np.ndarray, knobs: dict) -> np.ndarray:
    """Histogram of Oriented Gradients (Dalal & Triggs, 2005): a direction histogram per cell, normalised per block."""
    return hog(gray, feature_vector=True, **hog_params(knobs))


# ── 5b · LBP ───────────────── 👾 for every pixel: which of my neighbours are brighter than me?

def lbp_codes(gray: np.ndarray, knobs: dict) -> np.ndarray:
    """A texture code per pixel (Ojala, Pietikäinen & Harwood, 1994–96). "uniform" keeps P + 2 codes:
    0 = bright spot (every neighbour darker), P = flat or dark spot (every neighbour as bright or brighter:
    a flat patch counts as "brighter", so the masked black background lands here), 1 … P−1 = edges and
    corners, P + 1 = noisy, no clear pattern."""
    return local_binary_pattern(gray, knobs.get("lbp_points", 8), knobs.get("lbp_radius", 1), method="uniform")


def lbp_features(gray: np.ndarray, knobs: dict) -> np.ndarray:
    """One histogram of codes per cell of a grid × grid layout: what textures, and roughly where."""
    p, grid = knobs.get("lbp_points", 8), knobs.get("lbp_grid", 4)
    codes = lbp_codes(gray, knobs)
    bins, (h, w) = p + 2, gray.shape
    hists = []
    for gy in range(grid):
        for gx in range(grid):
            cell = codes[gy * h // grid:(gy + 1) * h // grid, gx * w // grid:(gx + 1) * w // grid]
            hist, _ = np.histogram(cell, bins=bins, range=(0, bins))
            hists.append(hist / max(cell.size, 1))
    return np.concatenate(hists)


# ── 5c · scaling ───────────── 👾 put every feature on the same ruler, measured on the TRAINING frames only

def fit_scaler(X_train: np.ndarray, knobs: dict):
    """standard: z = (x − μ) / σ · minmax: (x − min) / (max − min) · none: None. Fitted on train only: no peeking."""
    scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler()}.get(knobs.get("scaling") or "none")
    return scaler.fit(X_train) if scaler is not None else None


# ── 5d · augmentation ──────── 👾 a plausible *different* camera frame of the same situation

def flip(img: np.ndarray, horizontal: bool, vertical: bool) -> np.ndarray:
    if horizontal:
        img = cv2.flip(img, 1)                                   # mirror left ↔ right
    if vertical:
        img = cv2.flip(img, 0)                                   # mirror top ↔ bottom
    return img


def affine(img: np.ndarray, degrees: float, zoom: float) -> np.ndarray:
    """Rotate around the centre and zoom, in one 2 × 3 matrix (Euler's affine map). The edges are mirrored in."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, zoom)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def augment(img: np.ndarray, knobs: dict, rng: np.random.Generator) -> np.ndarray:
    """One random copy: maybe flipped, a little rotated and zoomed, a little brighter or darker."""
    img = flip(img, bool(knobs.get("augment_flip_horizontal")) and rng.random() < 0.5,
               bool(knobs.get("augment_flip_vertical")) and rng.random() < 0.5)
    lo, hi = knobs.get("augment_scale") or [1.0, 1.0]
    r = knobs.get("augment_rotate_degrees") or 0
    img = affine(img, rng.uniform(-r, r), rng.uniform(lo, hi))
    b = knobs.get("augment_brightness") or 0
    if b:
        img = brightness(img, 1 + rng.uniform(-b, b))
    return img


if __name__ == "__main__":  # ▶ this recipe on every frame, then the random forest takes the exam
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run_pipeline import run
    from stages.sandbox import refresh

    refresh("segmenting")                                   # stages 1–4 again, if you changed one of them
    ok = run("extracting", "random_forest", keep_going=True)
    print("\n👀 Every frame: build/extracting/preview.png · the exam: build/random_forest/preview.png")
    raise SystemExit(0 if ok else 1)
