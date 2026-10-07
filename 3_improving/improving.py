"""
🔆 Stage 3 · Improving — night shift on the ward: will we still see anything in dim light?
═════════════════════════════════════════════════════════════════════════════════════════
Lecture: Image Preprocessing Methods · Thu 24.09 · Why this stage exists: README.md · The values: action.yaml

This is the file CI runs for this stage. Press ▶ in PyCharm (or `python 3_improving/improving.py`):

    build/improving/report.md          the values you passed in, every number explained, the gates
    build/improving/before_after.png   your image for analyzing and one test frame per class: in → enhanced → Canny edges

Change one value in action.yaml, press ▶ again, and the report says what changed since your last run.

👾 A dim frame uses only a narrow band of the 256 grey levels: everything is murky. Improving spreads the
   brightness out so the structure is easy to see, on the brightness channel only, so colours don't shift:
     equalize  make every grey level equally common: the histogram becomes flat (one rule for the whole frame)
     clahe     equalize every tile of the frame separately, but clip each tile's histogram first so it doesn't
               blow up the noise (Zuiderveld, 1994). Dim corners get their own rule.
     stretch   map the darkest and brightest few percent to black and white, everything in between linearly
     gamma     bend the curve: γ < 1 brightens the shadows, γ > 1 darkens them
   Then the edges: Sobel measures how fast brightness changes in x and y; Canny (1986) thins that to one-pixel
   edges and keeps weak edges only when they touch strong ones. pass_on: edges hands those to Segmenting instead.

   A real startup argument: stage 5's HOG normalises contrast in every block itself, so the classifier may
   barely need this stage. Does that make the contrast gate wrong, or does it protect against a darker ward or a
   new camera that the test set doesn't show? Gates are decisions, not laws of nature.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # 🔒 finds stages/

import time

import cv2
import numpy as np

from stages.common import on_luma, to_luma
from stages.stage import Stage, picture, run   # 🔒 the plumbing: values, input, saving, report, gates

KEY = "improving"


def enhance(img: np.ndarray, stage_values: dict, color_space: str = "gray") -> np.ndarray:
    method = stage_values.get("method", "clahe")

    def fn(ch: np.ndarray) -> np.ndarray:
        if method in (None, "none"):
            return ch
        if method == "equalize":
            return cv2.equalizeHist(ch)
        if method == "clahe":
            t = int(stage_values.get("clahe_tile", 8))
            return cv2.createCLAHE(clipLimit=float(stage_values.get("clahe_clip", 2.0)), tileGridSize=(t, t)).apply(ch)
        if method == "stretch":
            lo, hi = np.percentile(ch, stage_values.get("stretch_percentiles") or [1, 99])
            return np.clip((ch.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)
        if method == "gamma":
            lut = (255.0 * (np.arange(256) / 255.0) ** float(stage_values.get("gamma", 1.0))).astype(np.uint8)
            return cv2.LUT(ch, lut)
        raise SystemExit(f"❌ Unknown method '{method}'. Choose none | equalize | clahe | stretch | gamma")

    return on_luma(img, color_space, fn)


def edges(img: np.ndarray, stage_values: dict, color_space: str = "gray") -> np.ndarray:
    return cv2.Canny(to_luma(img, color_space), stage_values.get("canny_low", 50), stage_values.get("canny_high", 150))


def improve(img: np.ndarray, stage_values: dict, space: str, steps: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One frame in → (enhanced, its Canny edges)."""
    picture(steps, "in: from stage 2", img, space)
    better = enhance(img, stage_values, space)
    picture(steps, str(stage_values.get("method")), better, space)
    edge_map = edges(better, stage_values, space)
    picture(steps, "Canny edges", edge_map)
    return better, edge_map


def contrast(img: np.ndarray, color_space: str = "gray") -> float:
    """Standard deviation of the brightness: how spread out the grey levels are."""
    return float(to_luma(img, color_space).std())


def main() -> None:
    stage = Stage(KEY)                           # the values from action.yaml
    stage_values, frames = stage.values, stage.frames()      # every frame stage 2 passed on, and your image for analyzing
    space, pass_on = frames.color_space, stage_values.get("pass_on", "enhanced")

    t0 = time.perf_counter()
    results = [improve(im, stage_values, space, stage.steps(i)) for i, im in enumerate(frames.images)]
    ms = 1000 * (time.perf_counter() - t0) / max(1, len(results))
    enhanced, edge_maps = [r[0] for r in results], [r[1] for r in results]
    if pass_on == "edges":
        stage.save(edge_maps, {"color_space": "gray"})
    else:
        stage.save(enhanced)

    before = np.array([contrast(im, space) for im in frames.data()])
    after = np.array([contrast(im, space) for im in frames.data(enhanced)])
    gate = stage_values.get("gate_min_dim_contrast")
    stage.metric("method", stage_values.get("method"))
    stage.metric("contrast before (σ, mean)", before.mean(), "Spread of the grey levels, averaged over the frames as they came in.")
    stage.metric("contrast after (σ, mean)", after.mean(), "The same after improving. Higher = easier to see structure (and noise).")
    stage.metric("contrast of dimmest 5% before", np.percentile(before, 5), "The night shift: the murkiest frames as they came in.")
    dim_after = stage.metric("contrast of dimmest 5% after", np.percentile(after, 5),
                             f"The murkiest frames after improving. The 🚦 gate wants at least {gate}.",
                             "good" if gate is None or np.percentile(after, 5) >= gate else "bad")
    stage.metric("edge pixels %", 100 * np.mean([(e > 0).mean() for e in frames.data(edge_maps)]),
                 "Share of pixels Canny calls an edge. A few percent outlines the chamber; much more is noise.")
    stage.metric("passed on", pass_on, "What stage 4 receives (pass_on).")
    stage.perf("ms per frame (enhance + Canny)", ms)
    stage.note("HOG normalises contrast per block too, so better-looking frames don't always mean better accuracy.")

    if (i := frames.snap_idx) is not None:
        stage.snap("contrast before", contrast(frames.images[i], space))
        stage.snap("contrast after", contrast(enhanced[i], space))
        stage.snap("edge pixels %", 100 * (edge_maps[i] > 0).mean())
        stage.snap_image("output", enhanced[i], space)
        stage.snap_image("edges", edge_maps[i], "gray")

    stage.tip("`method: \"equalize\"` against `\"clahe\"`: look at the dim frames in before_after.png. One rule for the frame, or one per tile?")
    stage.tip("`clahe_clip` at 1, 2 and 4: more contrast, and more noise. Where's the sweet spot?")
    stage.tip("`method: \"none\"`: the contrast gate fails, but does the test accuracy drop? Then argue about the gate (see the comment at the top of improving.py).")

    stage.gate("contrast of dimmest 5% frames", dim_after, min=gate)
    stage.finish(f"3 · Improving — {stage_values.get('method')} → passing on: {pass_on}")


if __name__ == "__main__":
    run(KEY, main)
