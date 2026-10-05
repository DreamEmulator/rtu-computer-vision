"""
🧬 Stage 5c · Scaling & Normalization Sandbox
═════════════════════════════════════════════
Every real frame in, one ruler out, with the same recipe.fit_scaler() CI fits on the training frames.

    📥 in   every frame, as stage 4 · Segmenting passed it on, described by 5a + 5b (recipe.extract())
    👀 look build/sandbox/stage_5c_look.png        features before | after | what the models think of it
    📝 log  build/sandbox/<unix time>_results_stage_5c.md

Press ▶ in PyCharm (pick "5c · Scaling" in the run menu), or from the repository root:

    python 04_feature-extraction/extracting_describe/sandboxes/sandbox_5c_scaling.py

👾 Centimetres and kilometres. HOG numbers run from 0 to about 0.3, LBP shares from 0 to 1. A model
   that measures distances between frames (kNN, an SVM) hears the features with the biggest numbers
   shouting and the rest whispering. Scaling puts every feature on the same ruler:
     min-max   (x − min) / (max − min)   squeezed into 0 … 1
     z-score   (x − μ) / σ               "how many standard deviations from an average frame" (Gauss)
   The ruler is measured on the TRAINING frames only. Measuring it on the test set would be peeking.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "run_pipeline.py").exists())))
from stages import sandbox as sb  # 🔒 the plumbing every sandbox shares (it checks your packages first)

import numpy as np

from stages.common import knobs_of
from stages.s5_extracting import feature_names

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#    🎛️  TINKER ZONE — change one thing, press ▶, compare the two results files
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
TRY = {}         # empty = exactly what CI does, with the knobs in ../action.yaml. Then try ONE of these:
#   "scaling": "minmax"     every feature squeezed into 0 … 1 on the training frames
#   "scaling": "standard"   every feature as a z-score: 0 = average, ±1 = one standard deviation
#   "scaling": "none"       the raw numbers: watch who shouts
#   "features": ["hog", "lbp", "histogram"]   add a feature with a different ruler, and scale again
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "extracting", "5c"
NAMES = ["scaling", "features"]   # this sub-step's knobs
LOUD = 0.10                       # "the loudest features": the widest 10 %


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)

    # 1. 📥 Describe every frame stage 4 passed on (no augmented copies here: those come in 5d)
    sb.refresh("segmenting")
    src = sb.load_images("segmenting")
    X, ms = sb.timed(lambda: np.array([np.concatenate(list(recipe.extract(sb.to_luma(img, src.color_space), knobs).values()))
                                       for img in src.images], dtype=np.float32))
    split = np.array([r["split"] for r in src.rows])
    tr, te = split == "train", split == "test"
    y = np.array([src.classes.index(r["label"]) if r["label"] in src.classes else -1 for r in src.rows])
    names = [n for n, _ in feature_names(knobs, src.images[0].shape)]
    names = names if len(names) == X.shape[1] else [f"feature {i}" for i in range(X.shape[1])]

    # 2. 📏 Measure the ruler on the training frames, then apply it to every frame: the recipe
    scaler = recipe.fit_scaler(X[tr], knobs)
    Xs = scaler.transform(X) if scaler is not None else X

    # 3. 📣 Who shouts? The share of a typical distance between two frames that the widest 10 % of features cause
    spread = X[tr].std(axis=0)
    loud = np.argsort(spread)[::-1][:max(1, int(LOUD * X.shape[1]))]
    rng = np.random.default_rng(0)
    pairs = rng.choice(np.flatnonzero(tr), size=(500, 2))

    def loud_share(M):
        d2 = (M[pairs[:, 0]] - M[pairs[:, 1]]) ** 2
        return 100 * d2[:, loud].sum() / max(d2.sum(), 1e-12)

    before, after = loud_share(X), loud_share(Xs)
    alive = np.flatnonzero(spread > 1e-9)
    picks = [loud[0], alive[np.argsort(spread[alive])[len(alive) // 2]], alive[np.argmin(spread[alive])]]
    ruler = (spread[alive].max() / spread[alive].min()) if len(alive) else 1.0
    outside = 100 * float(((Xs[te] < 0) | (Xs[te] > 1)).mean()) if knobs["scaling"] == "minmax" else None

    # 4. 🤖 What the models think: kNN measures distances, the forest only asks "bigger than?"
    rf_knobs, models = knobs_of("random_forest"), sb.recipe("random_forest")
    seed = knobs_of("digital_data").get("seed", 42)
    scores = {}
    for how in ("none", "minmax", "standard"):
        s = recipe.fit_scaler(X[tr], {**knobs, "scaling": how})
        M = s.transform(X) if s is not None else X
        knn = models.build_model({**rf_knobs, "model": "knn"}, seed).fit(M[tr], y[tr])
        scores[how] = float((knn.predict(M[te]) == y[te]).mean())
    forest = {}
    for how in ("none", knobs["scaling"]):
        s = recipe.fit_scaler(X[tr], {**knobs, "scaling": how})
        M = s.transform(X) if s is not None else X
        model = models.build_model({**rf_knobs, "model": "random_forest"}, seed).fit(M[tr], y[tr])
        forest[how] = float((model.predict(M[te]) == y[te]).mean())

    # 5. 👀 Three features before and after, and the models' verdicts
    fig = sb.new_figure(13, 3.9)
    a, b, c = fig.subplots(1, 3)
    short = [names[j].replace("HOG ", "").replace("LBP ", "LBP ") for j in picks]
    for ax, M, title in ((a, X, "before: raw numbers"), (b, Xs, f"after: {knobs['scaling']}")):
        for row, j in enumerate(picks, 1):                               # min … max, the middle half, the median
            v = M[tr][:, j]
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([v.min(), v.max()], [row, row], color="#90a4ae", lw=1.5)
            ax.plot([q1, q3], [row, row], color="#2e7d32", lw=9, solid_capstyle="butt")
            ax.plot([med], [row], "o", color="white", ms=4)
        ax.set_ylim(0.4, 3.6)
        ax.set_yticks([1, 2, 3], [f"widest\n{short[0]}", f"typical\n{short[1]}", f"narrowest\n{short[2]}"], fontsize=7)
        ax.set_title(title, fontsize=10)
    hows = list(scores)
    c.bar([f"kNN\n{h}" for h in hows], [scores[h] for h in hows],
          color=["#2e7d32" if h == knobs["scaling"] else "#90a4ae" for h in hows])
    c.bar([f"forest\n{h}" for h in forest], list(forest.values()), color="#8d6e63")
    c.set_ylim(0, 1); c.set_ylabel("test accuracy"); c.set_title("distances care, trees don't", fontsize=10)
    for i, v in enumerate(list(scores.values()) + list(forest.values())):
        c.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    look = sb.look("stage_5c_look.png", [(f"{X.shape[1]} features x {int(tr.sum())} training frames", sb.figure(fig))])

    # 6. 🚦 Judge the numbers
    metrics = [
        ("📐", "Rulers before scaling", f"widest feature {ruler:.0f}× wider than the narrowest", "check" if ruler > 10 else "info",
         "Spread (σ) of the widest feature over the narrowest one that isn't constant, on the training frames."),
        ("📣", "The loudest 10 % of features", f"{before:.0f} % → {after:.0f} % of a distance",
         "good" if after < before - 5 or knobs["scaling"] != "none" else "check",
         "Their share of the distance between two random frames, before → after scaling. Lower = the rest get a say too. "
         "With equal rulers each 10 % of the features would cause roughly 10 %."),
        ("👥", "kNN test accuracy", " · ".join(f"{h} {v:.2f}" for h, v in scores.items()), "info",
         "kNN decides by distance, so the ruler matters. (No augmented copies here, so CI's numbers differ a little.)"),
        ("🌳", "Forest test accuracy", " · ".join(f"{h} {v:.2f}" for h, v in forest.items()), "info",
         "A tree only asks 'is this feature bigger than t?'. Stretching a ruler moves t along with it: same answer."),
    ]
    if outside is not None:
        metrics.append(("🚪", "Test values outside 0 … 1", f"{outside:.2f} %", "check" if outside > 0 else "good",
                        "Min-max was measured on the training frames, so a test frame brighter than any of them lands above 1. "
                        "That's correct (no peeking), and a reason to prefer z-scores when the real world can surprise you."))
    metrics.append(("⏱️", "Time to describe every frame", f"{ms / 1000:.1f} s for {len(X)} frames", "info",
                    "5a + 5b on every frame. Scaling itself is two quick passes over the table."))

    tips = ['Set `"scaling": "none"` and watch the loudest features take over the distance.',
            "Why are the forest's two bars the same? Explain it in one sentence; it's an exam question.",
            "Look at the narrowest feature after z-scores: almost always 0, so one frame with a little of it lands far "
            "out. Z-scores make rare things loud. Is that what you want for a drop that's only in a few frames?",
            'Scaling matters most when features have different rulers: add `"histogram"` to `"features"` and compare.',
            "In 6 · Random forest, try `model: \"knn\"` with and without scaling: same lesson, in CI.",
            "Next: ▶ `sandbox_5d_augmentation.py`: more training frames, made up from the ones we have."]

    story = (f"👾 I described all **{len(X)}** frames with `recipe.extract()` ({' + '.join(knobs['features'])}: "
             f"{X.shape[1]} numbers each), measured the ruler on the **{int(tr.sum())}** training frames with "
             f"`recipe.fit_scaler()` (**{knobs['scaling']}**) and applied it to every frame. Before, the widest 10 % "
             f"of the features caused {before:.0f} % of the distance between two frames; after, {after:.0f} %. "
             f"kNN scored {scores['none']:.2f} unscaled and {scores[knobs['scaling']] if knobs['scaling'] in scores else scores['none']:.2f} "
             f"with {knobs['scaling']}.")
    log = sb.write_results(KEY, STEP, "🧬 Stage 5c · Scaling & Normalization", story,
                           files=[("📥 input", "build/segmenting/images/"), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips)
    sb.report(metrics, [look], log, knobs, defaults, NAMES)


if __name__ == "__main__":
    main()
