"""
🌳 Stage 6c · The Verdict Sandbox
═════════════════════════════════
One real frame in, a verdict out: how CI's forest votes on it (recipe.votes()), and where in the frame it looks.

    📥 in   build/extracting/features.npz          stage 5's vectors, and your snap's
    👀 look build/sandbox/stage_6c_look.png        where the forest looks | how the trees voted
    📝 log  build/sandbox/<unix time>_results_stage_6c.md

Press ▶ in PyCharm (pick "6c · The verdict" in the run menu), or from the repository root:

    python 05_random-forests/classifying_random-forest/sandboxes/sandbox_6c_verdict.py
    python 05_random-forests/classifying_random-forest/sandboxes/sandbox_6c_verdict.py low_fluid

👾 Election night. Every tree walks your frame down its own questions (6a) and picks a class; the forest
   averages their opinions. A landslide means every kind of tree agrees; a close race means your frame
   sits on the border between two classes. The heat map shows which parts of the frame the forest's
   questions are about, over all trees (Breiman's feature importance: how much each feature cleaned up
   the groups it split). If it glows on the wall instead of the chamber, the forest learned the wall.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "run_pipeline.py").exists())))
from stages import sandbox as sb  # 🔒 the plumbing every sandbox shares (it checks your packages first)

import cv2
import numpy as np

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#    🎛️  TINKER ZONE — change one thing, press ▶, compare the two results files
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
FRAME = "snap"   # "snap" = your Step 0 photo · "drop", "no_drop", "low_fluid" = a test frame of that class
TRY = {}         # empty = exactly the forest CI trains, with the knobs in ../action.yaml. Then try ONE of these:
#   "n_estimators": 5         a tiny electorate: is the vote still clear?
#   "max_features": None      trees that look at every feature: does the heat map get narrower?
#   "class_weight": "balanced"  rare classes count more when the trees learn: does your verdict change?
# Better than CI? Copy the value into the TINKER ZONE of ../action.yaml yourself: that's your decision.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ end of TINKER ZONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY, STEP = "random_forest", "6c"
NAMES = ["model", "n_estimators", "max_features", "class_weight"]   # the knobs that change the vote most


def main() -> None:
    recipe = sb.recipe(KEY)
    knobs, defaults = sb.knobs(KEY, TRY)

    # 1. 📥 Load stage 5's features, and the frame to judge
    data = sb.features()
    x, frame, true, caption = sb.pick(data, FRAME)
    k = len(data.classes)

    # 2. 🌳 Train the model CI trains, and ask it about this frame
    model = recipe.build_model(knobs, data.seed).fit(data.X_train, data.y_train)
    proba = model.predict_proba(x[None])[0]
    verdict = int(np.argmax(proba))
    order = np.argsort(proba)[::-1]
    v = recipe.votes(model, x, k)                                        # None for svm and knn

    # 3. 🔥 Where the forest looks: every feature's importance spread over its region of the frame
    s = sb.zoom(frame.shape)
    vis = sb.bigger(frame, data.space)
    heat_share, top = None, []
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        heat = np.zeros(frame.shape[:2], np.float64)
        for i, (_, box) in enumerate(data.names):
            if box:
                y0, x0, y1, x1 = box
                heat[y0:y1, x0:x1] += imp[i] / ((y1 - y0) * (x1 - x0))  # importance per pixel
        heat_share = 100 * sum(imp[i] for i, (_, box) in enumerate(data.names) if box) / max(imp.sum(), 1e-12)
        colour = cv2.applyColorMap(np.uint8(255 * heat / max(heat.max(), 1e-12)), cv2.COLORMAP_INFERNO)
        vis = cv2.addWeighted(vis, 0.45, cv2.resize(colour, vis.shape[1::-1], interpolation=cv2.INTER_NEAREST), 0.55, 0)
        top = np.argsort(imp)[::-1][:5]

    # 4. 🗳️ The election result
    fig = sb.new_figure(4.6, 3.6)
    ax = fig.add_subplot()
    bars = v if v is not None else proba
    ax.barh(data.classes, bars, color=["#2e7d32" if i == verdict else "#bdbdbd" for i in range(k)])
    for i, b in enumerate(bars):
        ax.text(b, i, f" {b}" if v is not None else f" {b:.0%}", va="center", fontsize=10)
    ax.set_xlim(0, (bars.max() or 1) * 1.25)
    ax.set_title("trees voting for each class" if v is not None else f"{knobs['model']}: probability per class", fontsize=10)
    fig.tight_layout()
    look = sb.look("stage_6c_look.png", [("where the forest looks" if top is not None and len(top) else caption, vis),
                                         (f"verdict: {data.classes[verdict]}", sb.figure(fig))])

    # 5. 🚦 Judge the numbers
    margin = proba[order[0]] - proba[order[1]]
    right = None if true is None else verdict == true
    metrics = [
        ("🏛️", "Verdict", data.classes[verdict] + ("" if right is None else (" ✅" if right else f" ❌ (truth: {data.classes[true]})")),
         "info" if right is None else "good" if right else "bad", f"What the forest says about {caption}."),
        ("🗳️", "Votes for it", f"{v[verdict]} of {v.sum()} trees" if v is not None else f"{proba[verdict]:.0%}",
         "good" if proba[verdict] >= 0.7 else "check",
         "A landslide (70 % or more) means trees with different questions agree. Below that, the frame is a border case."),
        ("🥈", "Runner-up", f"{data.classes[order[1]]}, {margin:.0%} behind", "check" if margin < 0.2 else "info",
         "A close race. What in this frame could look like the runner-up?"),
    ]
    if heat_share is not None:
        metrics.append(("🗺️", "Importance with a place", f"{heat_share:.0f} %", "info",
                        "Share of the forest's attention that sits in a region of the frame (HOG, LBP, pixels). "
                        "The rest belongs to features about the whole frame, like the brightness histogram."))

    lines = []
    if len(top):
        lines = ["## 🔎 The five features the forest relies on most", "",
                 "| # | Feature | Importance | This frame |", "|---|---|---|---|"]
        lines += [f"| {r + 1} | {data.names[i][0]} | {model.feature_importances_[i]:.3f} | {x[i]:+.2f} |" for r, i in enumerate(top)]
        lines += ["", "*This frame*: stage 5's number after `scaling`, so 0 is an average frame and +1 one standard deviation above."]

    tips = ["Does the heat map glow on the drip chamber, or on the wall around it? A forest that learned the wall "
            "is right for the wrong reason (❓ How to train our model on cows?).",
            'Another frame: `FRAME = "low_fluid"`. Same forest, different election.']
    if margin < 0.2:
        tips.insert(0, "A close race: open the look picture of 4a–4d for this frame. What made it ambiguous?")
    tips.append("Next: ▶ `sandbox_6d_exam.py`: not one frame but the whole locked test set, and the gates.")

    story = (f"👾 I trained the forest CI trains and asked it about **{caption}**. "
             + (f"**{v[verdict]} of {v.sum()} trees** voted **{data.classes[verdict]}**, " if v is not None
                else f"`{knobs['model']}` says **{data.classes[verdict]}** ")
             + f"with **{data.classes[order[1]]}** {margin:.0%} behind."
             + ("" if right is None else (" That's right. ✅" if right else f" Wrong: it's {data.classes[true]}. ❌")))
    log = sb.write_results(KEY, STEP, "🌳 Stage 6c · The Verdict", story,
                           files=[("📥 input", "build/extracting/features.npz"), ("👀 look", sb.rel(look))],
                           used=knobs, defaults=defaults, names=NAMES, metrics=metrics, tips=tips, extra="\n".join(lines))
    sb.report(metrics, [look], log, knobs, defaults, NAMES)


if __name__ == "__main__":
    main()
