"""
🌳 Stage 6d · The Exam Sandbox
══════════════════════════════
The whole locked test set in, a verdict on the MODEL out: CI's own exam, cross_validation() and grade().

    📥 in   build/extracting/features.npz          stage 5's vectors: training and the locked test set
    👀 look build/sandbox/stage_6d_look.png        confusion matrix · recall per class · the frames it got wrong
    📝 log  build/sandbox/<unix time>_results_stage_6d.md

Press ▶ in PyCharm (pick "6d · The exam" in the run menu), or from the repository root:

    python 05_random-forests/classifying_random-forest/sandboxes/sandbox_6d_exam.py

👾 The exam. The model studied the training frames; now it sits a test it has never seen. The exam
   lives in 🔒 stages/s6_classifying.py, not in recipe.py, on purpose: you change the model, never the
   exam. Two numbers decide whether CI ships it (the 🚦 gates in ../action.yaml): overall accuracy, and
   the recall of the worst class. Recall of low_fluid: of all the chambers that were really running dry,
   how many did we catch? That one is patient safety.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "run_pipeline.py").exists())))
from stages import sandbox as sb  # 🔒 the plumbing every sandbox shares (it checks your packages first)

import numpy as np

from stages.s6_classifying import cross_validation, grade, plot_results  # 🔒 CI's exam, the same for every model

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#    🎛️  TINKER ZONE — change one thing, press ▶, compare the two results files
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
TRY = {}         # empty = exactly CI's exam, with the knobs in ../action.yaml. Then try ONE of these:
#   "model": "knn"                 another student sits the same exam
#   "class_weight": "balanced"     does the worst class's recall go up? What does it cost the others?
#   "cv_folds": 10                 a practice exam in 10 parts instead of 5: does the estimate get steadier?
#   "gate_min_class_recall": 0.9   a stricter promise to the hospital. Would we still ship?
# Better than CI? Copy the value into ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "random_forest", "6d"
NAMES = ["model", "class_weight", "cv_folds", "gate_min_accuracy", "gate_min_class_recall"]


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)

    # 1. 📥 Load stage 5's features
    data = sb.features()
    k = len(data.classes)

    # 2. 📝 The practice exam: cross-validation on the training frames only
    model = recipe.build_model(knobs, data.seed)
    cv = cross_validation(model, data.X_train, data.y_train, data.g_train, knobs["cv_folds"], data.seed)

    # 3. 🎓 The real exam: train on everything we may learn from, then the locked test set
    model.fit(data.X_train, data.y_train)
    y_pred = model.predict(data.X_test)
    exam = grade(data.y_test, y_pred, k)
    acc, recalls, cm = exam["accuracy"], exam["recalls"], exam["confusion"]
    baseline = np.bincount(data.y_train).argmax()
    baseline_acc = float((data.y_test == baseline).mean())

    # 4. 👀 CI's own picture of the exam
    look = sb.OUT / "stage_6d_look.png"
    sb.OUT.mkdir(parents=True, exist_ok=True)
    plot_results(look, cm, data.classes, np.flatnonzero(y_pred != data.y_test), data.I_test, data.y_test, y_pred,
                 None, data.space, f"6d · The exam — {knobs['model']}")

    # 5. 🚦 Judge the model, the way the gates will
    gate_acc, gate_rec = knobs["gate_min_accuracy"], knobs["gate_min_class_recall"]
    worst = int(np.argmin(recalls))
    ships = (gate_acc is None or acc >= gate_acc) and (gate_rec is None or recalls[worst] >= gate_rec)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    t, p = np.unravel_index(np.argmax(off), off.shape)
    metrics = [
        ("🎓", "Test accuracy", f"{acc:.2f}", "good" if gate_acc is None or acc >= gate_acc else "bad",
         f"Share of the {len(data.y_test)} locked test frames it got right. 🚦 Gate: at least {gate_acc}."),
        ("🧯", "Worst-class recall", f"{recalls[worst]:.2f} ({data.classes[worst]})",
         "good" if gate_rec is None or recalls[worst] >= gate_rec else "bad",
         f"Of all the {data.classes[worst]} frames, how many it caught. 🚦 Gate: at least {gate_rec}. "
         "Accuracy can look fine while one class is mostly missed."),
        ("🎲", "Baseline", f"{baseline_acc:.2f}", "info",
         f'Always answering "{data.classes[baseline]}". Anything near this hasn\'t learned a thing.'),
        ("📝", "Practice exam (cv)", f"{cv.mean():.2f} ± {cv.std():.2f}" if cv is not None else "off (cv_folds < 2)",
         "info" if cv is None else "check" if abs(cv.mean() - acc) > 0.1 else "good",
         "Cross-validation on the training frames. Close to the test accuracy = a trustworthy estimate. "
         "Far apart = the test set is small, or the model got lucky (or unlucky)."),
        ("🔀", "Most common mistake", f"{data.classes[t]} called {data.classes[p]} ({off[t, p]}×)" if off.any() else "none",
         "check" if off.any() else "good",
         "Not all mistakes cost the same. Calling a dry chamber 'drop' lets it run dry; the opposite is a false alarm."),
        ("⚖️", "F1 (macro)", f"{exam['f1 (macro)']:.2f}", "info",
         "Precision and recall in one number, averaged over the classes so a small class counts as much as a big one."),
    ]

    tips = []
    if not ships:
        tips.append("CI wouldn't ship this. Which gate failed? Fix that one first: 6b for the forest, or an earlier stage.")
    if recalls[worst] < 0.9:
        tips.append(f'Try `"class_weight": "balanced"`: the trees take {data.classes[worst]} more seriously. Watch the other recalls.')
    tips += ['Same exam, another model: `"model": "knn"`, then `"svm"`. Who passes?',
             "Open the frames it got wrong in the look picture. Would *you* have got them right? What would you need to see?",
             "Like what you see? Put the values in ../action.yaml and ▶ `recipe.py`: CI's stage, same exam, same gates."]

    story = (f"👾 I trained `{knobs['model']}` on **{len(data.y_train)}** training frames and gave it CI's exam: "
             f"**{len(data.y_test)}** locked test frames it had never seen. It got **{acc:.0%}** right "
             f"(always guessing {data.classes[baseline]} would get {baseline_acc:.0%}). Its weakest class is "
             f"**{data.classes[worst]}**, caught {recalls[worst]:.0%} of the time. "
             + ("**CI would ship it.** 🚢" if ships else "**CI would not ship it.** 🛑"))
    log = sb.write_results(KEY, STEP, "🌳 Stage 6d · The Exam", story,
                           files=[("📥 input", "build/extracting/features.npz"), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips)
    sb.report(metrics, [look], log, knobs, defaults, NAMES)


if __name__ == "__main__":
    main()
