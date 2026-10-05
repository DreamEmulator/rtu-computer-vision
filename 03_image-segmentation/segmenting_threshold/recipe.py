"""
✂️ Stage 4 · Segmenting — the recipe
════════════════════════════════════
The maths of this stage and nothing else: one frame in, one mask out. CI runs exactly this file:

    .github/workflows/pipeline.yml
      └─ segmenting_threshold/action.yaml    🎛️ knobs + 🚦 gates
           └─ stages/s4_segmenting.py         🔒 plumbing: every frame, previews, metrics, gates
                └─ recipe.py                  🧪 you are here

The sandboxes next to this file call these same functions on one real frame, one sub-step at a time:

    4a threshold() → 4b morphology() → 4c keep_contours() → 4d boundary() → apply()

Every function gets `knobs`: the TINKER ZONE of action.yaml (plus a sandbox's TRY).
Change a knob there; change the *algorithm* here. Add a method of your own, say a "kmeans"
threshold, and let the gates tell you whether it's good enough to ship.

▶ Press ▶ on this file: your recipe runs on every frame (what CI does), then Extracting and the
  random forest, so you see what your change does to the test accuracy.
"""
from __future__ import annotations

import cv2
import numpy as np

SHAPES = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}
MORPH = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE, "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}
MODES = {"external": cv2.RETR_EXTERNAL, "list": cv2.RETR_LIST, "ccomp": cv2.RETR_CCOMP, "tree": cv2.RETR_TREE}
POINTS = {"none": cv2.CHAIN_APPROX_NONE, "simple": cv2.CHAIN_APPROX_SIMPLE}


def segment(gray: np.ndarray, knobs: dict) -> tuple[np.ndarray, list, float]:
    """The whole stage for one grey frame → (mask, kept contours, threshold used)."""
    mask, t = threshold(gray, knobs)
    mask, contours = keep_contours(morphology(mask, knobs), knobs)
    return boundary(mask, knobs), contours, t


# ── 4a · threshold ─────────── 👾 sort every pixel into object (white, 255) or background (black, 0)

def threshold(gray: np.ndarray, knobs: dict) -> tuple[np.ndarray, float]:
    """Returns (binary mask 0/255, threshold value used; nan when every neighbourhood has its own)."""
    method = knobs.get("threshold", "otsu")
    mode = cv2.THRESH_BINARY_INV if knobs.get("invert", True) else cv2.THRESH_BINARY
    if method == "global":
        t, mask = cv2.threshold(gray, knobs.get("global_value", 127), 255, mode)       # one rule for everyone
    elif method == "otsu":
        t, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)                  # Otsu picks T (the 0 is ignored)
    elif method == "triangle":
        t, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_TRIANGLE)              # Zack et al. pick T
    elif method == "adaptive":
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode,  # compare with the local mean − C
                                     odd(max(3, knobs.get("adaptive_block", 31))), knobs.get("adaptive_c", 5))
        t = float("nan")
    else:
        raise SystemExit(f"❌ Unknown threshold '{method}'. Choose global | otsu | triangle | adaptive")
    return mask, float(t)


# ── 4b · morphology ────────── 👾 plane the splinters off and fill the cracks, like a woodworker

def morphology(mask: np.ndarray, knobs: dict) -> np.ndarray:
    kernel = tool(knobs.get("morph_shape", "ellipse"), knobs.get("morph_kernel", 3))
    for op in knobs.get("morphology") or []:
        if op not in MORPH:
            raise SystemExit(f"❌ Unknown morphology step '{op}'. Choose erode | dilate | open | close")
        mask = cv2.morphologyEx(mask, MORPH[op], kernel, iterations=int(knobs.get("morph_iterations") or 1))
    return mask


# ── 4c · contours ──────────── 👾 trace every blob like your hand on paper, keep the big ones

def find_contours(mask: np.ndarray, knobs: dict) -> tuple[list, np.ndarray | None]:
    """Every outline in the mask (Suzuki & Abe, 1985) → (contours, hierarchy)."""
    mode, points = knobs.get("contour_mode", "external"), knobs.get("contour_points", "simple")
    if mode not in MODES or points not in POINTS:
        raise SystemExit("❌ contour_mode must be external | list | ccomp | tree, contour_points none | simple")
    contours, hierarchy = cv2.findContours(mask, MODES[mode], POINTS[points])
    return list(contours), hierarchy


def keep_contours(mask: np.ndarray, knobs: dict) -> tuple[np.ndarray, list]:
    """Throw away outlines smaller than min_contour_area → (mask of the objects we keep, their contours)."""
    contours, _ = find_contours(mask, knobs)
    kept = [k for k in contours if cv2.contourArea(k) >= knobs.get("min_contour_area", 8)]
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, kept, -1, 255, thickness=cv2.FILLED)
    return (filled if knobs.get("fill_holes") else cv2.bitwise_and(mask, filled)), kept


# ── 4d · boundary ──────────── 👾 scoop out the cookie, keep the crust: β(A) = A − (A ⊖ B)

def boundary(mask: np.ndarray, knobs: dict) -> np.ndarray:
    method = knobs.get("boundary") or "none"
    if method == "none":
        return mask                                                            # pass the whole objects on
    kernel = tool(knobs.get("boundary_shape", "rect"), knobs.get("boundary_kernel", 3))
    if method == "inner":
        return cv2.subtract(mask, cv2.erode(mask, kernel))                     # the cookie minus a shrunk cookie
    if method == "outer":
        return cv2.subtract(cv2.dilate(mask, kernel), mask)                    # a grown cookie minus the cookie
    if method == "gradient":
        return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)             # grown minus shrunk
    raise SystemExit(f"❌ Unknown boundary '{method}'. Choose none | inner | outer | gradient")


# ── → stage 5 ──────────────── what Extracting receives

def apply(img: np.ndarray, mask: np.ndarray, contours: list, how: str) -> np.ndarray:
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


# ── helpers ───────────────────

def odd(k: int) -> int:
    """Most OpenCV kernels need an odd size so they have a centre pixel."""
    k = max(1, int(k))
    return k if k % 2 else k + 1


def tool(shape: str, size: int) -> np.ndarray:
    """The structuring element: the shape morphology slides over the mask."""
    if shape not in SHAPES:
        raise SystemExit(f"❌ Unknown shape '{shape}'. Choose ellipse | rect | cross")
    k = odd(size)
    return cv2.getStructuringElement(SHAPES[shape], (k, k))


if __name__ == "__main__":  # ▶ this recipe on every frame, then Extracting and the random forest
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from run_pipeline import run
    from stages.sandbox import refresh

    refresh("improving")                                    # stages 1–3 again, if you changed one of them
    ok = run("segmenting", "random_forest", keep_going=True)
    print("\n👀 Every frame: build/segmenting/preview.png · the model: build/random_forest/preview.png")
    raise SystemExit(0 if ok else 1)
