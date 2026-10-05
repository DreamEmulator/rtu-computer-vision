"""
🌳 Stage 6 · Classifying — the random forest recipe
═══════════════════════════════════════════════════
The model and nothing else: which classifier, built how. CI runs exactly this file:

    .github/workflows/pipeline.yml
      └─ classifying_random-forest/action.yaml   🎛️ knobs + 🚦 gates
           └─ stages/s6_classifying.py            🔒 plumbing: load the features, train, the exam, gates
                └─ recipe.py                      🧪 you are here

The sandboxes next to this file call these same functions on the real features from stage 5:

    6a grow_tree() → 6b build_model() → 6c votes() → 6d the exam (🔒 plumbing: you don't write your own exam)

Every function gets `knobs`: the TINKER ZONE of action.yaml (plus a sandbox's TRY).
Change a knob there; change the *model* here. Try another scikit-learn classifier, say
"gradient_boosting", and let the gates tell you whether it's good enough to ship.

▶ Press ▶ on this file: your recipe trains on every frame and takes the exam on the locked test set.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ── 6a · one tree ──────────── 👾 twenty questions: every split asks "is this number bigger than that?"

def tree_knobs(knobs: dict) -> dict:
    """The knobs every tree gets, alone (6a) or in a forest (6b)."""
    return {"max_depth": knobs.get("max_depth"), "min_samples_leaf": knobs.get("min_samples_leaf", 1),
            "class_weight": knobs.get("class_weight")}


def grow_tree(knobs: dict, seed: int) -> DecisionTreeClassifier:
    """One decision tree (Breiman et al., 1984). It sees every frame and every feature: nothing random."""
    return DecisionTreeClassifier(**tree_knobs(knobs), random_state=seed)


# ── 6b · the forest ────────── 👾 many trees, each on a random sample of frames and features, then a vote

def build_model(knobs: dict, seed: int):
    name = knobs.get("model", "random_forest")
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=knobs.get("n_estimators", 200),
                                      max_features=knobs.get("max_features", "sqrt"),   # Ho's random subspaces
                                      **tree_knobs(knobs), random_state=seed, n_jobs=-1)  # bootstrap: Efron
    if name == "svm":  # an SVM gives distances, not probabilities: Platt (1999) calibrates them on held-out folds
        svm = SVC(C=knobs.get("svm_c", 1.0), kernel=knobs.get("svm_kernel", "rbf"), gamma=knobs.get("svm_gamma", "scale"))
        return CalibratedClassifierCV(svm, ensemble=False)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=knobs.get("knn_neighbors", 5))
    raise SystemExit(f"❌ Unknown model '{name}'. Choose random_forest | svm | knn")


# ── 6c · the vote ──────────── 👾 every tree gives its opinion on one frame; the forest averages them

def votes(model, x: np.ndarray, n_classes: int) -> np.ndarray | None:
    """How many trees pick each class for one feature vector. None for models without trees (svm, knn).

    The forest's own answer averages the trees' probabilities (a soft vote, scikit-learn's way). With fully
    grown trees every tree is 100 % sure of its pick, so the soft vote and this head count agree.
    """
    if not hasattr(model, "estimators_"):
        return None
    picks = [int(model.classes_[int(tree.predict(x[None])[0])]) for tree in model.estimators_]
    return np.bincount(picks, minlength=n_classes)


if __name__ == "__main__":  # ▶ this recipe on every frame: train, then the exam on the locked test set
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from run_pipeline import run
    from stages.sandbox import refresh

    refresh("extracting")                                   # stages 1–5 again, if you changed one of them
    ok = run("random_forest", "random_forest", keep_going=True)
    print("\n👀 The exam: build/random_forest/preview.png")
    raise SystemExit(0 if ok else 1)
