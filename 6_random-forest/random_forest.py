"""
🌳 Stage 6 · Classifying · Random Forest — is the infusion running, stopped, or about to run dry?
════════════════════════════════════════════════════════════════════════════════════════════════
Lecture: Image Classification with Random Forests · Tue 06.10 · Why: README.md · The values: action.yaml

This is the file CI runs for this stage. Press ▶ in PyCharm (or `python 6_random-forest/random_forest.py`):

    build/random_forest/report.md          the values you passed in, every number explained, the gates
    build/random_forest/before_after.png   the confusion matrix · lone trees vs the forest · the frames it got wrong

Change one value in action.yaml, press ▶ again, and the report says what changed since your last run.

👾 Twenty questions. One decision tree learns to guess the class by asking yes/no questions about stage 5's
   numbers: "is the HOG edge in cell (4,4) stronger than 0.21?" (Breiman et al., 1984). It picks the question
   that splits the training frames into the purest groups, then asks again in each group. Left alone it keeps
   asking until every group is pure: 100 % on the frames it learned from, much less on new ones. It memorised.

   The wisdom of crowds. In 1907 Francis Galton asked 787 fairgoers to guess the weight of an ox; the middle
   guess was within 1 % of the truth. A forest grows many trees, each from a random sample of the frames
   (Efron's bootstrap, 1979), each question looking at a random handful of features (max_features; Ho, 1995),
   and lets them vote (Breiman, 2001). The trees make different mistakes, so the mistakes cancel out. That
   only works if they ARE different: let every tree see every feature (max_features: null) and they grow alike.

   The model is yours to change (build_model below); the exam isn't (🔒 stages/exam.py, the same for every model).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # 🔒 finds stages/

import time

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from stages import exam                         # 🔒 the exam: the same for every model
from stages.stage import Stage, load_code, run  # 🔒 the plumbing: values, report, gates

KEY = "random_forest"


# ── The model ─────────────────────────────────────────────────────────────────────────────────────────
#    random_forest  many trees, a vote (above)
#    svm            the widest possible street between the classes; the kernel trick bends it through curved
#                   space (Cortes & Vapnik, 1995). It gives distances, not probabilities: Platt (1999) turns
#                   them into probabilities on held-out folds
#    knn            you are what your neighbours are (Fix & Hodges, 1951). It keeps every training frame
#    SVM and kNN measure distances, so they need stage 5's scaling. Trees only ask "bigger than?" and don't.

def build_model(stage_values: dict, seed: int):
    name = stage_values.get("model", "random_forest")
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=stage_values.get("n_estimators", 200),
                                      max_features=stage_values.get("max_features", "sqrt"),
                                      max_depth=stage_values.get("max_depth"), min_samples_leaf=stage_values.get("min_samples_leaf", 1),
                                      class_weight=stage_values.get("class_weight"), random_state=seed, n_jobs=-1)
    if name == "svm":
        svm = SVC(C=stage_values.get("svm_c", 1.0), kernel=stage_values.get("svm_kernel", "rbf"), gamma=stage_values.get("svm_gamma", "scale"))
        return CalibratedClassifierCV(svm, ensemble=False)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=stage_values.get("knn_neighbors", 5))
    raise SystemExit(f"❌ Unknown model '{name}'. Choose random_forest | svm | knn")


def main() -> None:
    stage = Stage(KEY)                           # the values from action.yaml
    stage_values = stage.values
    seed = stage.all_values["digital_data"].get("seed", 42)
    d = exam.load_features()                     # stage 5's numbers: X_train, X_test, X_snap and the labels

    # 📝 The practice exam: cross-validation on the training frames only
    model = build_model(stage_values, seed)
    scores = exam.cross_validation(model, d.X_train, d.y_train, d.g_train, stage_values.get("cv_folds"), seed)
    if scores is not None:
        stage.metric("cv accuracy (mean ± std)", f"{scores.mean():.3f} ± {scores.std():.3f}",
                     "The practice exam on the training frames. Close to the test accuracy = a trustworthy estimate.")

    # 🌳 Learn from every training frame, then answer the locked test set and your image for analyzing
    t0 = time.time()
    model.fit(d.X_train, d.y_train)
    train_s = time.time() - t0
    t1 = time.perf_counter()
    y_pred = model.predict(d.X_test)
    predict_ms = 1000 * (time.perf_counter() - t1) / len(d.y_test)
    snap_proba = model.predict_proba(d.X_snap)[0] if len(d.X_snap) else None
    joblib.dump(model, stage.dir / "model.joblib", compress=3)
    stage.metric("model", stage_values.get("model", "random_forest"))

    side = None
    if hasattr(model, "estimators_"):            # a forest: the crowd against its members
        lone, crowd = wisdom(model, d.X_test, d.y_test)
        stage.metric("average lone tree", float(lone.mean()), "Each tree on its own, on the test set. Compare with the forest's test accuracy.")
        stage.metric("wisdom of the crowd", float(crowd[-1] - lone.mean()),
                     "Forest minus the average tree. The vote fixes mistakes the trees don't share; near 0 means they all "
                     "make the same mistakes (try max_features).", "good" if crowd[-1] - lone.mean() > 0.02 else "check")
        names = load_code("extracting").feature_names(stage.all_values["extracting"], d.I_test.shape[1:])
        if len(names) == len(model.feature_importances_):
            top = np.argsort(model.feature_importances_)[::-1][:3]
            stage.metric("features it relies on most", [names[i][0] for i in top],
                         "Where in the frame the forest's questions are about. On the drip chamber, or on the wall?")
        if snap_proba is not None:
            k = int(np.argmax(snap_proba))
            votes = sum(int(t.predict(d.X_snap[:1])[0]) == k for t in model.estimators_)
            stage.snap("trees voting for it", f"{votes} of {len(model.estimators_)}")
        side = lambda ax: draw_wisdom(ax, lone, crowd)  # noqa: E731

    # 🎓 The exam (🔒 the same for every model)
    exam.take(stage, d, y_pred, snap_proba, stage.dir / "model.joblib", train_s, predict_ms, side,
              f"6 · {stage_values.get('model', 'random_forest').replace('_', ' ').title()}")

    stage.tip("`n_estimators: \"5\"`, then 200, then 500. Where does the accuracy stop paying for the time?")
    stage.tip("`max_features: \"null\"`: every tree may look at every feature. Watch the wisdom of the crowd shrink.")
    stage.tip("`model: \"svm\"` and `\"knn\"`, then `scaling: \"none\"` in 5_extracting. Why do these two suffer and the forest doesn't?")
    stage.tip("`class_weight: \"balanced\"`: does the recall of the worst class go up? What does it cost the others?")
    stage.finish(f"6 · {stage_values.get('model', 'random_forest')}")


def wisdom(model, X_test: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """→ (every tree's own test accuracy, the forest's accuracy with the first 1, 2, … n trees voting)."""
    P = np.stack([tree.predict_proba(X_test) for tree in model.estimators_])        # trees × frames × classes
    lone = (P.argmax(axis=2) == y_test).mean(axis=1)
    crowd = (np.cumsum(P, axis=0).argmax(axis=2) == y_test).mean(axis=1)
    return lone, crowd


def draw_wisdom(ax, lone: np.ndarray, crowd: np.ndarray) -> None:
    ax.plot(np.arange(1, len(crowd) + 1), crowd, color="#2e7d32", label="the forest")
    ax.axhline(lone.mean(), color="#999", ls="--", label=f"average lone tree {lone.mean():.2f}")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("trees voting"); ax.set_ylabel("test accuracy"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("the crowd, one tree at a time", fontsize=10)


if __name__ == "__main__":
    run(KEY, main)
