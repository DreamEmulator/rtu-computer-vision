"""
🌳 Stage 6b · The Forest Sandbox
════════════════════════════════
The real features in, a forest out, with the same recipe.build_model() CI trains.

    📥 in   build/extracting/features.npz          stage 5's vectors
    👀 look build/sandbox/stage_6b_look.png        every lone tree's score vs the forest's | the score as trees join
    📝 log  build/sandbox/<unix time>_results_stage_6b.md

Press ▶ in PyCharm (pick "6b · The forest" in the run menu), or from the repository root:

    python 05_random-forests/classifying_random-forest/sandboxes/sandbox_6b_forest.py

👾 The wisdom of crowds. In 1907 Francis Galton asked 787 fairgoers to guess the weight of an ox;
   the middle guess was within 1 % of the truth. One tree from 6a is a rough guesser. The forest grows
   many, each from a random sample of the frames (Efron's bootstrap, 1979) and allowed to look at only a
   random handful of features per question (Ho, 1995). The trees disagree in different places, so
   their mistakes cancel out when they vote (Breiman, 2001). That only works if they're different.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "run_pipeline.py").exists())))
from stages import sandbox as sb  # 🔒 the plumbing every sandbox shares (it checks your packages first)

import joblib
import numpy as np

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#    🎛️  TINKER ZONE — change one thing, press ▶, compare the two results files
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
TRY = {}         # empty = exactly the forest CI trains, with the knobs in ../action.yaml. Then try ONE of these:
#   "n_estimators": 10        how many trees vote: 1, 10, 200, 1000
#   "max_features": None      None = every question may look at every feature: the trees grow alike
#                             "sqrt" = √1924 ≈ 44 random features per question: the trees grow different
#   "max_depth": 3            shallow trees: weak guessers on their own. Does the crowd still win?
#   "model": "svm"            no trees at all: compare its score with the forest's
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "random_forest", "6b"
NAMES = ["model", "n_estimators", "max_features", "max_depth", "min_samples_leaf", "class_weight"]


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)

    # 1. 📥 Load stage 5's features
    data = sb.features()

    # 2. 🌳 Train the model CI trains
    model, ms = sb.timed(lambda: recipe.build_model(knobs, data.seed).fit(data.X_train, data.y_train))
    acc = float((model.predict(data.X_test) == data.y_test).mean())
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    size_kb = buffer.tell() / 1024

    if not hasattr(model, "estimators_"):                        # svm or knn: no trees, no crowd
        fig = sb.new_figure(6, 2.4)
        fig.text(0.5, 0.5, f"{knobs['model']}: test accuracy {acc:.2f}\nno trees, so no crowd to watch",
                 ha="center", va="center", fontsize=13)
        look = sb.look("stage_6b_look.png", [(knobs["model"], sb.figure(fig))])
        metrics = [("🧪", "Test accuracy", f"{acc:.2f}", "info", "On the locked test set."),
                   ("⏱️", "Time to train", f"{ms:.0f} ms", "info", "On all training frames."),
                   ("📦", "Model size", f"{size_kb:.0f} KB", "info", "What a camera would have to store.")]
        story = f"👾 `{knobs['model']}` has no trees, so there's no vote to watch. It scores **{acc:.2f}** on the test set."
        tips = ['Set `TRY = {"model": "random_forest"}` and compare.']
    else:
        # 3. 🗳️ Ask every tree on its own, then the crowd as it grows tree by tree
        P = np.stack([tree.predict_proba(data.X_test) for tree in model.estimators_])   # trees × frames × classes
        lone = (P.argmax(axis=2) == data.y_test).mean(axis=1)                            # each tree's own score
        crowd = (np.cumsum(P, axis=0).argmax(axis=2) == data.y_test).mean(axis=1)        # first k trees together
        picks = P.argmax(axis=2)
        n = len(model.estimators_)
        same = sum((np.sum(picks == k, axis=0) * (np.sum(picks == k, axis=0) - 1)).sum() for k in range(len(data.classes)))
        alike = same / max(1, n * (n - 1) * picks.shape[1])                              # P(two trees agree on a frame)
        enough = int(np.argmax(crowd >= acc - 0.005)) + 1

        # 4. 👀 Lone trees vs the crowd, and the crowd as it grows
        fig = sb.new_figure(10, 3.8)
        a, b = fig.subplots(1, 2)
        a.hist(lone, bins=np.linspace(0, 1, 41), color="#8bc34a")
        a.axvline(lone.mean(), color="#555", ls="--", label=f"average tree {lone.mean():.2f}")
        a.axvline(acc, color="#2e7d32", lw=3, label=f"the forest {acc:.2f}")
        a.set_xlabel("test accuracy"); a.set_ylabel("trees"); a.set_title(f"{n} trees on their own", fontsize=10)
        a.legend(fontsize=8, loc="upper left")
        b.plot(np.arange(1, n + 1), crowd, color="#2e7d32")
        b.set_xscale("log"); b.set_ylim(0, 1.02)
        b.set_xlabel("trees voting"); b.set_ylabel("test accuracy"); b.set_title("the crowd, one tree at a time", fontsize=10)
        fig.tight_layout()
        look = sb.look("stage_6b_look.png", [(f"lone trees vs the forest ({n} trees)", sb.figure(fig))])

        # 5. 🚦 Judge the numbers
        gain = acc - lone.mean()
        metrics = [
            ("🌳", "Forest test accuracy", f"{acc:.2f}", "good" if acc >= knobs["gate_min_accuracy"] else "check",
             f"All {n} trees voting, on the locked test set. CI ships from {knobs['gate_min_accuracy']:.2f}."),
            ("🌱", "Average lone tree", f"{lone.mean():.2f}", "info",
             f"Each tree alone (worst {lone.min():.2f}, best {lone.max():.2f}). Compare with your tree from 6a."),
            ("🧠", "Wisdom of the crowd", f"{gain:+.2f}", "good" if gain > 0.02 else "check",
             "Forest minus average tree. Higher is better: the vote fixes mistakes the trees don't share. "
             "Near 0 means the trees all make the same mistakes."),
            ("👯", "How alike the trees are", f"{100 * alike:.0f} %", "info",
             "How often two random trees give the same answer on a test frame. Lower = more different = a wiser crowd. "
             "max_features is the knob."),
            ("📈", "Trees needed", f"{enough} of {n}", "info",
             "After this many trees the forest is within half a percent of its final score. The rest cost time, not accuracy."),
            ("⏱️", "Time to train", f"{ms:.0f} ms", "info", "All trees, spread over your CPU cores (n_jobs=-1)."),
            ("📦", "Model size", f"{size_kb:.0f} KB", "info", "Grows with the number of trees and their depth."),
        ]
        tips = []
        if gain <= 0.02:
            tips.append('The crowd barely beats one tree: are the trees too alike? Check `"max_features"`.')
        tips += ['`"max_features": None`: every tree may look at every feature. Watch "How alike" go up and the wisdom go down.',
                 f'`"n_estimators": {max(1, enough)}`: about the same score for less time. Where would you put it on a clip-on camera?',
                 '`"max_depth": 3`: weak trees. Is a crowd of weak but different guessers still wise?',
                 "Next: ▶ `sandbox_6c_verdict.py`: how the trees vote on your snap, and where in the frame they look."]
        story = (f"👾 I trained the forest CI trains with `recipe.build_model()`: **{n} trees**, each on a bootstrap "
                 f"sample of the training frames, each question looking at `max_features = {knobs['max_features']}`. "
                 f"On their own the trees score **{lone.mean():.2f}** on average. Voting together: **{acc:.2f}**. "
                 f"Two trees agree on a test frame {100 * alike:.0f} % of the time.")

    log = sb.write_results(KEY, STEP, "🌳 Stage 6b · The Forest", story,
                           files=[("📥 input", "build/extracting/features.npz"), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips)
    sb.report(metrics, [look], log, knobs, defaults, NAMES)


if __name__ == "__main__":
    main()
