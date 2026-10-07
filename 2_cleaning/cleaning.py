"""
🧽 Stage 2 · Cleaning — our clip-on camera is cheap. Can we trust its pixels?
═════════════════════════════════════════════════════════════════════════════
Lecture: Image Preprocessing Methods · Thu 24.09 · Why this stage exists: README.md · The values: action.yaml

This is the file CI runs for this stage. Press ▶ in PyCharm (or `python 2_cleaning/cleaning.py`):

    build/cleaning/report.md          the values you passed in, every number explained, the gates
    build/cleaning/before_after.png   your image for analyzing and one test frame per class: in → each filter → what it removed

Change one value in action.yaml, press ▶ again, and the report says what changed since your last run.

👾 A cheap sensor adds noise: grain everywhere (Gaussian noise) and the odd dead pixel stuck at black or white
   (salt and pepper). Cleaning replaces every pixel by something computed from its neighbourhood:
     gaussian   a weighted average, the weights a bell curve (Gauss). Smooths grain; smears a dead pixel into a blob.
     median     the middle value of the neighbourhood. One extreme value can't move a median, so dead pixels
                vanish, and edges stay sharp (Tukey, 1970s).
     bilateral  an average that only counts neighbours of similar brightness: smooths flat areas, keeps edges.
     nlmeans    averages patches that look alike, wherever they are in the frame (Buades et al., 2005).
   Every filter trades noise for detail. The drop is only a few pixels wide at 128 × 128: a big kernel cleans
   the wall and erases the drop. So the report shows both numbers: noise σ, and sharpness.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # 🔒 finds stages/

import time

import cv2
import numpy as np

from stages.common import odd
from stages.stage import Stage, picture, run   # 🔒 the plumbing: values, input, saving, report, gates

KEY = "cleaning"


def clean(img: np.ndarray, stage_values: dict, space: str = "gray", steps: list | None = None) -> np.ndarray:
    """One frame in, one cleaned frame out. `method` may be a list: the filters run one after the other."""
    picture(steps, "in: from stage 1", img, space)
    original = img
    methods = stage_values.get("method", "median")
    for method in methods if isinstance(methods, list) else [methods]:
        k = odd(stage_values.get("kernel", 3))
        if method in (None, "none"):
            continue
        elif method == "gaussian":
            img = cv2.GaussianBlur(img, (k, k), stage_values.get("sigma") or 0)
        elif method == "median":
            img = cv2.medianBlur(img, max(3, k))
        elif method == "bilateral":
            img = cv2.bilateralFilter(img, stage_values.get("bilateral_diameter", 7),
                                      stage_values.get("bilateral_sigma_color", 50), stage_values.get("bilateral_sigma_space", 50))
        elif method == "nlmeans":
            img = cv2.fastNlMeansDenoising(img, None, stage_values.get("nlmeans_h", 10), 7, 21)
        else:
            raise SystemExit(f"❌ Unknown method '{method}'. Choose none | gaussian | median | bilateral | nlmeans")
        picture(steps, f"{method} {k}×{k}" if method in ("gaussian", "median") else method, img, space)
    picture(steps, "removed (×4)", removed(original, img), space)
    return img


# ── How we measure it ─────────────────────────────────────────────────────────────────────────────────

_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)  # difference of two Laplacians


def estimate_noise(img: np.ndarray) -> float:
    """Immerkær (1996): a fast estimate of Gaussian noise σ that ignores smooth image structure."""
    if img.ndim == 3:
        return float(np.mean([estimate_noise(img[..., k]) for k in range(img.shape[2])]))
    h, w = img.shape
    response = cv2.filter2D(img.astype(np.float64), -1, _NOISE_KERNEL)[1:-1, 1:-1]
    return float(np.sum(np.abs(response)) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2)))


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian: high for crisp edges (and, sadly, for noise)."""
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def removed(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """What the filter took away, 4× brighter so you can see it. Grain is fine; the outline of the drop is not."""
    return np.clip(cv2.absdiff(before, after).astype(np.int32) * 4, 0, 255).astype(np.uint8)


def main() -> None:
    stage = Stage(KEY)                           # the values from action.yaml
    stage_values, frames = stage.values, stage.frames()      # every frame stage 1 passed on, and your image for analyzing
    space = frames.color_space

    t0 = time.perf_counter()
    cleaned = [clean(im, stage_values, space, stage.steps(i)) for i, im in enumerate(frames.images)]
    ms = 1000 * (time.perf_counter() - t0) / max(1, len(cleaned))
    stage.save(cleaned)

    noise_before = np.mean([estimate_noise(im) for im in frames.data()])
    noise_after = np.mean([estimate_noise(im) for im in frames.data(cleaned)])
    sharp_before = np.mean([sharpness(im) for im in frames.data()])
    sharp_after = np.mean([sharpness(im) for im in frames.data(cleaned)])
    gate = stage_values.get("gate_max_noise")
    stage.metric("method", stage_values.get("method"))
    stage.metric("noise σ before", noise_before, "Estimated grain in the frames as they came in (Immerkær's method).")
    stage.metric("noise σ after", noise_after, f"What's left after cleaning. Lower is better; the 🚦 gate allows {gate}.",
                 "good" if gate is None or noise_after <= gate else "bad")
    stage.metric("sharpness before", sharp_before, "Variance of the Laplacian: crisp edges score high (and so does noise).")
    stage.metric("sharpness after", sharp_after, "Some loss is the price of less noise. Losing most of it means the drop is blurring away.",
                 "check" if sharp_after < 0.3 * sharp_before else "info")
    stage.perf("ms per frame", ms)
    stage.note("Denoising always trades noise for detail: watch both numbers, not just one.")

    if (i := frames.snap_idx) is not None:
        stage.snap("noise σ before", estimate_noise(frames.images[i]))
        stage.snap("noise σ after", estimate_noise(cleaned[i]))
        stage.snap_image("output", cleaned[i], space)
        stage.snap_image("removed", removed(frames.images[i], cleaned[i]), space)

    stage.tip("`method: \"gaussian\"` against `\"median\"` at the same `kernel`. Look at the dead pixels in before_after.png: "
              "why does the median delete them while the Gaussian only smears them?")
    stage.tip("Find the smallest `kernel` that passes the noise gate for each method. Which keeps more sharpness?")
    stage.tip("`method: \"none\"`: which stage breaks, and does stage 6 even get to run?")

    stage.gate("noise σ after cleaning", noise_after, max=gate)
    stage.finish(f"2 · Cleaning — {stage_values.get('method')} (kernel {odd(stage_values.get('kernel', 3))})")


if __name__ == "__main__":
    run(KEY, main)
