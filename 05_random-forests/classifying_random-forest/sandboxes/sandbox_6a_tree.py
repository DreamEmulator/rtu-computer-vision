"""
🌳 Stage 6a · One Decision Tree Sandbox
═══════════════════════════════════════
The real features in, one tree out, with the same recipe.grow_tree() every tree in CI's forest is made of.

    📥 in   build/extracting/features.npz          stage 5's vectors: 1 frame = ~2000 numbers
    👀 look build/sandbox/stage_6a_look.png        the questions it asks your frame | the top of the tree
    📝 log  build/sandbox/<unix time>_results_stage_6a.md   with the tree's rules in plain text

Press ▶ in PyCharm (pick "6a · One tree" in the run menu), or from the repository root:

    python 05_random-forests/classifying_random-forest/sandboxes/sandbox_6a_tree.py
    python 05_random-forests/classifying_random-forest/sandboxes/sandbox_6a_tree.py low_fluid

👾 Twenty questions. The Robot learns to guess the class by asking yes/no questions about the numbers
   from stage 5: "is the HOG edge in cell (4,4) stronger than 0.21?" It picks the question that splits
   the training frames into the purest groups (Breiman et al., 1984), then asks again in each group,
   until every group holds one class. Your frame walks down the tree, one answer at a time.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "run_pipeline.py").exists())))
from stages import sandbox as sb  # 🔒 the plumbing every sandbox shares (it checks your packages first)

import cv2
import numpy as np
from sklearn.tree import export_text, plot_tree

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#    🎛️  TINKER ZONE — change one thing, press ▶, compare the two results files
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
FRAME = "snap"   # "snap" = your Step 0 photo · "drop", "no_drop", "low_fluid" = a test frame of that class
TRY = {}         # empty = exactly the trees CI grows, with the knobs in ../action.yaml. Then try ONE of these:
#   "max_depth": 3            at most 3 questions per frame: a simpler tree. null = ask until every group is pure
#   "min_samples_leaf": 10    every final group must hold at least 10 frames: no rules for one odd frame
#   "class_weight": "balanced"  rare classes count more when the tree picks its questions
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "random_forest", "6a"
NAMES = ["max_depth", "min_samples_leaf", "class_weight"]   # this sub-step's knobs


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)

    # 1. 📥 Load stage 5's features, and the frame to ask about
    data = sb.features()
    x, frame, true, caption = sb.pick(data, FRAME)
    names = [n for n, _ in data.names]

    # 2. 🌱 Grow one tree on the training frames
    tree, ms = sb.timed(lambda: recipe.grow_tree(knobs, data.seed).fit(data.X_train, data.y_train))
    train_acc = float((tree.predict(data.X_train) == data.y_train).mean())
    test_acc = float((tree.predict(data.X_test) == data.y_test).mean())

    # 3. 🪜 Walk your frame down the tree: every node on its path is one question
    t = tree.tree_
    path = tree.decision_path(x[None]).indices
    questions = []
    for node in path[:-1]:                                         # the last node is the leaf: the answer
        f, thr = t.feature[node], t.threshold[node]
        questions.append((f, thr, bool(x[f] <= thr)))
    leaf = path[-1]
    verdict = int(np.argmax(t.value[leaf][0]))

    # 4. 👀 Number the questions on the frame (green = yes, orange = no) and draw the top of the tree
    s = sb.zoom(frame.shape)
    vis = sb.bigger(frame, data.space)
    marks = [(n, [v * s for v in data.names[f][1]], (90, 200, 90) if yes else (31, 140, 240))
             for n, (f, _, yes) in enumerate(questions, 1) if data.names[f][1]]
    for _, (y0, x0, y1, x1), colour in marks:                                   # boxes first, numbers on top
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), colour, 2)
    for n, (y0, x0, _, _), colour in marks:
        cv2.putText(vis, str(n), (x0 + 4, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    fig = sb.new_figure(9, 4.6)
    plot_tree(tree, max_depth=2, feature_names=names, class_names=data.classes, filled=True, impurity=False,
              proportion=True, fontsize=7, ax=fig.add_subplot())
    fig.tight_layout()
    look = sb.look("stage_6a_look.png", [(f"{len(questions)} questions for {caption}", vis),
                                         ("the top of the tree", sb.figure(fig))])

    # 5. 🚦 Judge the numbers
    gap = train_acc - test_acc
    gate = knobs["gate_min_accuracy"]
    used = len({f for f in t.feature if f >= 0})
    metrics = [
        ("📚", "Training accuracy", f"{train_acc:.2f}", "check" if train_acc > 0.99 and gap > 0.1 else "info",
         "On the frames it learned from. 1.00 with a much lower test accuracy means it memorised them: "
         "every odd frame got its own rule."),
        ("🧪", "Test accuracy", f"{test_acc:.2f}", "good" if test_acc >= gate else "check",
         f"On the locked test set. One tree isn't gated, but for comparison: CI ships a model from {gate:.2f}. "
         "Hold on to this number for 6b."),
        ("🪜", "Depth", f"{tree.get_depth()}", "info", "The longest chain of questions. max_depth caps it."),
        ("🍂", "Leaves", f"{tree.get_n_leaves()}", "info",
         "Final groups. Many leaves for a few hundred frames means many rules for single frames."),
        ("🔎", "Features it uses", f"{used} of {len(names)}", "info",
         "A tree only asks about the features that split the classes best. All the others it ignores."),
        ("❓", "Questions for your frame", f"{len(questions)}", "info",
         f"The path of {caption} down the tree. Numbered in the look picture, in plain words below."),
        ("⏱️", "Time to grow", f"{ms:.0f} ms", "info", "One tree. A forest of 200 costs about 200 times as much (spread over your CPU cores)."),
    ]

    lines = ["## ❓ The questions it asked " + caption, "",
             "Values are stage 5's numbers after `scaling`: 0 is an average frame, +1 is one standard deviation above.", ""]
    for n, (f, thr, yes) in enumerate(questions, 1):
        lines.append(f"{n}. Is **{names[f]}** ≤ {thr:.2f}? This frame: {x[f]:.2f} → **{'yes' if yes else 'no'}**")
    lines += ["", f"→ The tree says **{data.classes[verdict]}**"
              + ("" if true is None else f" (truth: {data.classes[true]} {'✅' if verdict == true else '❌'})"), "",
              "## 🌳 The tree's rules (the first 3 levels)", "", "```text",
              export_text(tree, feature_names=names, class_names=data.classes, max_depth=3).rstrip(), "```"]

    tips = []
    if gap > 0.1:
        tips.append('It memorised: try `"max_depth": 3` or `"min_samples_leaf": 10`. Does the test accuracy go up or down?')
    tips += ["Look at the regions in the look picture. Does the tree ask about the drip chamber, or about the wall?",
             'Same tree, another frame: `FRAME = "low_fluid"`. Does it take a different path?',
             "Next: ▶ `sandbox_6b_forest.py` grows 200 of these trees, each a little different, and lets them vote."]

    story = (f"👾 I grew one tree with `recipe.grow_tree()` on **{len(data.y_train)}** training frames "
             f"(augmented copies included), each described by **{len(names)}** numbers. It asks up to "
             f"**{tree.get_depth()}** questions and ends in **{tree.get_n_leaves()}** leaves. It scores "
             f"**{train_acc:.2f}** on the frames it learned from and **{test_acc:.2f}** on the test set. "
             f"For {caption} it asked {len(questions)} questions and said **{data.classes[verdict]}**.")
    log = sb.write_results(KEY, STEP, "🌳 Stage 6a · One Decision Tree", story,
                           files=[("📥 input", "build/extracting/features.npz"), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips, extra="\n".join(lines))
    sb.report(metrics, [look], log, knobs, defaults, NAMES)


if __name__ == "__main__":
    main()
