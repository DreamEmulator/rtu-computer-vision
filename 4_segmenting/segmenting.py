"""
✂️ Stage 4 · Segmenting — which pixels belong to the drip chamber, and which are just background?
════════════════════════════════════════════════════════════════════════════════════════════════
Lecture: Image Segmentation · Tue 29.09 · Why this stage exists: README.md · The values: action.yaml

This is the file CI runs for this stage. Press ▶ in PyCharm (or run `python 4_segmenting/segmenting.py`)
and you get exactly what CI gets, on your machine:

    build/segmenting/report.md           the values you passed in, every number explained, the gates
    build/segmenting/before_after.png    your snap and one test frame per class, after every step

Change one value in action.yaml, press ▶ again, and the report says what changed since your last run.
Read this file top to bottom: four steps, one function each, then main() runs them on every frame.

    4a threshold()  →  4b morphology()  →  4c keep_contours()  →  4d boundary()  →  apply(): what stage 5 gets
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # 🔒 finds stages/

import time

import cv2
import numpy as np

from stages.common import to_luma
from stages.stage import Stage, picture, run   # 🔒 the plumbing: values, input, saving, report, gates


# ── 4a · Threshold ────────────────────────────────────────────────────────────────────────────────────
# 👾 Sort every pixel into one of two bins: object (white, 255) or background (black, 0).
#    The rule that decides is the threshold T: darker than T is object (invert: true, the chamber is darker
#    than the backlit wall), brighter is background. Four ways to pick T:
#      global    one T for every pixel, you choose it (global_value). Simple, and fragile: our backlight is
#                brighter in the middle than in the corners (vignetting), so one T is wrong somewhere.
#      otsu      try all 256 T's and keep the one that splits the brightness histogram into two groups that
#                are as different as possible (Otsu, 1979). Perfect when the histogram has two hills.
#      triangle  made for ONE big hill with a tail, like a mostly grey wall (Zack et al., 1977).
#      adaptive  a different T for every pixel: the Gaussian-weighted mean of its own neighbourhood
#                (adaptive_block pixels wide) minus adaptive_c. Uneven light stops mattering.

def threshold(gray: np.ndarray, stage_values: dict) -> tuple[np.ndarray, float]:
    """→ (mask of 0 and 255, the T it used; nan when every neighbourhood has its own)."""
    method = stage_values.get("threshold", "otsu")
    mode = cv2.THRESH_BINARY_INV if stage_values.get("invert", True) else cv2.THRESH_BINARY
    if method == "global":
        t, mask = cv2.threshold(gray, stage_values.get("global_value", 127), 255, mode)
    elif method == "otsu":
        t, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)                  # the 0 is ignored: Otsu picks T
    elif method == "triangle":
        t, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_TRIANGLE)
    elif method == "adaptive":
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode,
                                     odd(max(3, stage_values.get("adaptive_block", 31))), stage_values.get("adaptive_c", 5))
        t = float("nan")
    else:
        raise SystemExit(f"❌ Unknown threshold '{method}'. Choose global | otsu | triangle | adaptive")
    return mask, float(t)


# ── 4b · Morphology ───────────────────────────────────────────────────────────────────────────────────
# 👾 The mask from 4a is a rough plank: splinters (specks of noise) stick out, cracks and knotholes (holes)
#    run through it. Morphology (Matheron & Serra, 1964, to measure grains in iron ore) slides a small shape,
#    the structuring element, over the mask:
#      erode   plane a layer off every edge: splinters thinner than the tool vanish, the object shrinks
#      dilate  fill along every edge: cracks narrower than the tool close, the object grows
#      open    erode, then dilate: splinters gone, size kept      close  dilate, then erode: cracks gone, size kept
#    The order matters: [open, close] and [close, open] give different masks. Try both.

def morphology(mask: np.ndarray, stage_values: dict) -> np.ndarray:
    kernel = tool(stage_values.get("morph_shape", "ellipse"), stage_values.get("morph_kernel", 3))
    for op in stage_values.get("morphology") or []:
        if op not in MORPH:
            raise SystemExit(f"❌ Unknown morphology step '{op}'. Choose erode | dilate | open | close")
        mask = cv2.morphologyEx(mask, MORPH[op], kernel, iterations=int(stage_values.get("morph_iterations") or 1))
    return mask


# ── 4c · Contours ─────────────────────────────────────────────────────────────────────────────────────
# 👾 Trace every white blob like your hand on paper: follow the border pixel by pixel until you're back where
#    you started (Suzuki & Abe, 1985: that's cv2.findContours). Now we have objects instead of pixels, and we
#    can throw away the small ones (min_contour_area, in px²). Careful: the drop is small too.
#    fill_holes colours in everything inside a kept outline: every closed curve has an inside (Jordan, 1887).

def find_contours(mask: np.ndarray, stage_values: dict) -> list:
    mode, points = stage_values.get("contour_mode", "external"), stage_values.get("contour_points", "simple")
    if mode not in MODES or points not in POINTS:
        raise SystemExit("❌ contour_mode must be external | list | ccomp | tree, contour_points none | simple")
    contours, _ = cv2.findContours(mask, MODES[mode], POINTS[points])
    return list(contours)


def keep_contours(mask: np.ndarray, stage_values: dict) -> tuple[np.ndarray, list]:
    """→ (mask of the objects we keep, their outlines)."""
    kept = [k for k in find_contours(mask, stage_values) if cv2.contourArea(k) >= stage_values.get("min_contour_area", 8)]
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, kept, -1, 255, thickness=cv2.FILLED)
    return (filled if stage_values.get("fill_holes") else cv2.bitwise_and(mask, filled)), kept


# ── 4d · Boundary ─────────────────────────────────────────────────────────────────────────────────────
# 👾 Scoop out the cookie, keep the crust: β(A) = A − (A ⊖ B), the object minus a slightly eroded copy.
#    inner = the crust inside, outer = a ring just outside, gradient = both. none passes the whole objects on.
#    Contours (4c) give you the outline as points, good for measuring; a boundary gives it as pixels.

def boundary(mask: np.ndarray, stage_values: dict) -> np.ndarray:
    method = stage_values.get("boundary") or "none"
    if method == "none":
        return mask
    kernel = tool(stage_values.get("boundary_shape", "rect"), stage_values.get("boundary_kernel", 3))
    if method == "inner":
        return cv2.subtract(mask, cv2.erode(mask, kernel))
    if method == "outer":
        return cv2.subtract(cv2.dilate(mask, kernel), mask)
    if method == "gradient":
        return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    raise SystemExit(f"❌ Unknown boundary '{method}'. Choose none | inner | outer | gradient")


# ── → stage 5 · What Extracting receives ──────────────────────────────────────────────────────────────
#    masked    the frame, with everything outside the objects painted black (the default)
#    mask      only the shape: white objects on black
#    crop      a zoom on the biggest object, hopefully the chamber
#    original  the frame as it came in: the segmentation is ignored. That's an ablation: if the accuracy
#              at the end stays the same, this stage isn't helping (yet).

def apply(img: np.ndarray, mask: np.ndarray, contours: list, how: str) -> np.ndarray:
    if how == "original":
        return img
    if how == "mask":
        return mask
    if how == "masked":
        return cv2.bitwise_and(img, img, mask=mask)
    if how == "crop":
        if not contours:
            return img
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return cv2.resize(img[y:y + h, x:x + w], img.shape[1::-1], interpolation=cv2.INTER_AREA)
    raise SystemExit(f"❌ Unknown pass_on '{how}'. Choose masked | mask | crop | original")


# ── One frame, all steps ──────────────────────────────────────────────────────────────────────────────

def segment(img: np.ndarray, stage_values: dict, space: str, steps: list | None = None):
    """One frame in → (what stage 5 receives, the mask, the kept outlines, T). `steps` collects the pictures."""
    picture(steps, "in: from stage 3", img, space)
    mask, t = threshold(to_luma(img, space), stage_values)
    picture(steps, "4a threshold", mask)
    mask = morphology(mask, stage_values)
    picture(steps, "4b morphology", mask)
    mask, contours = keep_contours(mask, stage_values)
    picture(steps, "4c contours", overlay(img, mask, contours, space), "bgr")
    mask = boundary(mask, stage_values)
    if (stage_values.get("boundary") or "none") != "none":
        picture(steps, "4d boundary", mask)
    out = apply(img, mask, contours, stage_values.get("pass_on", "masked"))
    picture(steps, "out: to stage 5", out, "gray" if out.ndim == 2 else space)
    return out, mask, contours, t


def main() -> None:
    stage = Stage("segmenting")                  # the values from action.yaml
    stage_values, frames = stage.values, stage.frames()  # every frame stage 3 passed on, and your snap
    space, how = frames.color_space, stage_values.get("pass_on", "masked")

    outputs, masks, outlines, thresholds = [], [], [], []
    t0 = time.perf_counter()
    for i, img in enumerate(frames.images):
        out, mask, contours, t = segment(img, stage_values, space, stage.steps(i))
        outputs.append(out); masks.append(mask); outlines.append(contours); thresholds.append(t)
    ms = 1000 * (time.perf_counter() - t0) / len(frames.images)
    stage.save(outputs, {"color_space": "gray" if how == "mask" else space})

    # 📊 The numbers, over every frame of the dataset (your snap gets its own below)
    data = frames.data_idx
    fg = np.array([(masks[i] > 0).mean() for i in data])
    per_frame = float(np.mean([len(outlines[i]) for i in data]))
    stage.metric("threshold method", stage_values.get("threshold"))
    data_t = [thresholds[i] for i in data]
    if not np.all(np.isnan(data_t)):
        stage.metric("mean threshold value", float(np.nanmean(data_t)), "The average T it picked (0 = black, 255 = white).")
    stage.metric("foreground %", 100 * fg.mean(), "Share of the pixels kept as object. A drip chamber fills part of the frame, "
                 "not most of it: above 60 % check invert.", "good" if 0.5 < 100 * fg.mean() <= 60 else "check")
    stage.metric("contours per frame", per_frame, "Objects kept per frame. A handful is right; dozens are specks of noise.",
                 "good" if per_frame <= 10 else "check")
    stage.metric("empty masks %", 100 * (fg < 0.005).mean(), "Frames where nothing was kept: stage 5 gets a black frame.",
                 "good" if (fg < 0.005).mean() == 0 else "check")
    stage.metric("full masks %", 100 * (fg > 0.95).mean(), "Frames where everything was kept: the segmentation did nothing.",
                 "good" if (fg > 0.95).mean() == 0 else "check")
    stage.metric("passed on", how, "What stage 5 receives (pass_on).")
    if (stage_values.get("boundary") or "none") != "none":
        stage.metric("boundary", stage_values.get("boundary"))
    stage.perf("ms per frame", ms)

    if (i := frames.snap_idx) is not None:
        stage.snap("foreground %", 100 * (masks[i] > 0).mean())
        stage.snap("contours", len(outlines[i]))
        stage.snap_image("overlay", overlay(frames.images[i], masks[i], outlines[i], space), "bgr")
        stage.snap_image("output", outputs[i], "gray" if how == "mask" else space)

    # 💡 What to try next, depending on what happened
    if per_frame > 10:
        stage.tip("Lots of small objects: put `open` first in `morphology`, or raise `min_contour_area` (the drop is small, though).")
    if stage_values.get("threshold") in ("global", "otsu"):
        stage.tip("One T for the whole frame: compare with `threshold: adaptive` in the corners of before_after.png.")
    if 100 * fg.mean() > 60:
        stage.tip("Most pixels count as object: is `invert` right for your photos?")
    stage.tip("`pass_on: original` keeps this stage but ignores it. Same accuracy at the end? Then segmentation isn't helping yet.")

    stage.gate("empty mask ratio", (fg < 0.005).mean(), max=stage_values.get("gate_max_empty_ratio"))
    stage.gate("full mask ratio", (fg > 0.95).mean(), max=stage_values.get("gate_max_full_ratio"))
    stage.finish(f"4 · Segmenting — {stage_values.get('threshold')} threshold, morphology {stage_values.get('morphology')}")


# ── Helpers ───────────────────────────────────────────────────────────────────────────────────────────

SHAPES = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}
MORPH = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE, "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}
MODES = {"external": cv2.RETR_EXTERNAL, "list": cv2.RETR_LIST, "ccomp": cv2.RETR_CCOMP, "tree": cv2.RETR_TREE}
POINTS = {"none": cv2.CHAIN_APPROX_NONE, "simple": cv2.CHAIN_APPROX_SIMPLE}


def odd(k: int) -> int:
    """Most OpenCV kernels need an odd size, so they have a centre pixel."""
    k = max(1, int(k))
    return k if k % 2 else k + 1


def tool(shape: str, size: int) -> np.ndarray:
    """The structuring element: the shape morphology slides over the mask."""
    if shape not in SHAPES:
        raise SystemExit(f"❌ Unknown shape '{shape}'. Choose ellipse | rect | cross")
    return cv2.getStructuringElement(SHAPES[shape], (odd(size), odd(size)))


def overlay(img: np.ndarray, mask: np.ndarray, contours: list, space: str) -> np.ndarray:
    """The frame in grey, the mask in green, every kept outline in red."""
    vis = cv2.cvtColor(to_luma(img, space), cv2.COLOR_GRAY2BGR)
    tint = vis.copy()
    tint[mask > 0] = (60, 200, 60)
    vis = cv2.addWeighted(vis, 0.6, tint, 0.4, 0)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 1)
    return vis


if __name__ == "__main__":
    run("segmenting", main)
