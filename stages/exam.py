"""🔒 The exam: the same for every classifier, so nobody grades their own homework.

A classifying stage (6_random-forest/random_forest.py, 6_neural-network/neural_network.py) trains its model,
predicts the locked test set and your snap, then calls take(). That writes the numbers, the gates and
before_after.png: the confusion matrix, how the model learned, and the test frames it got wrong.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from stages.common import stage_dir, to_display


def load_features() -> SimpleNamespace:
    """Stage 5's output: X_* the feature vectors (for the forest), I_* the pixels (for the network), y_* the labels."""
    feats = stage_dir("extracting") / "features.npz"
    if not feats.exists():
        raise SystemExit("❌ No features from Extracting. Locally: python run_pipeline.py --to extracting")
    d = np.load(feats, allow_pickle=False)
    meta = json.loads((stage_dir("extracting") / "meta.json").read_text())
    return SimpleNamespace(**{k: d[k] for k in d.files if k != "classes"}, classes=[str(c) for c in d["classes"]],
                           color_space=meta.get("color_space", "gray"))


def cross_validation(model, X, y, groups, folds: int, seed: int) -> np.ndarray | None:
    """A practice exam on the TRAINING frames (Stone, 1974): train on k−1 folds, grade on the one left out, rotate.
    Augmented copies share a group with their original, so they never leak across folds. None when folds < 2."""
    if not folds or folds < 2:
        return None
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_score(model, X, y, groups=groups, cv=cv)


def grade(y_test, y_pred, n_classes: int) -> dict:
    """The exam on the locked test set."""
    labels = range(n_classes)
    return {"accuracy": accuracy_score(y_test, y_pred),
            "precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "recalls": recall_score(y_test, y_pred, average=None, labels=labels, zero_division=0),
            "confusion": confusion_matrix(y_test, y_pred, labels=labels)}


def take(stage, d: SimpleNamespace, y_pred: np.ndarray, snap_proba, model_file, train_s: float, predict_ms: float,
         side=None, title: str = "") -> dict:
    """Grade the predictions, write the report's numbers and gates, and draw before_after.png. → the grades."""
    stage_values, classes = stage.values, d.classes
    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(d.y_train), 1)), d.y_train)
    exam = grade(d.y_test, y_pred, len(classes))
    acc, recalls, cm = exam["accuracy"], exam["recalls"], exam["confusion"]
    (stage.dir / "labels.json").write_text(json.dumps(classes))
    plot(stage.dir / "before_after.png", cm, classes, np.flatnonzero(y_pred != d.y_test), d.I_test, d.y_test, y_pred,
         side, d.color_space, title)

    gate_acc, gate_rec = stage_values.get("gate_min_accuracy"), stage_values.get("gate_min_class_recall")
    base = baseline.score(np.zeros((len(d.y_test), 1)), d.y_test)
    worst = int(np.argmin(recalls))
    stage.metric("baseline (always guess the majority)", base, "What a model that learned nothing scores. Anything near it hasn't learned.")
    stage.metric("test accuracy", acc, f"Share of the locked test frames it got right. 🚦 Gate: at least {gate_acc}.",
                 "good" if gate_acc is None or acc >= gate_acc else "bad")
    for k in ("precision (macro)", "recall (macro)", "f1 (macro)"):
        stage.metric(k, exam[k])
    stage.metric("recall per class", {k: float(r) for k, r in zip(classes, recalls)},
                 f"Of all frames of a class, how many it caught. Worst: {classes[worst]}. 🚦 Gate: at least {gate_rec}. "
                 "Missing a chamber that runs dry is the dangerous mistake.",
                 "good" if gate_rec is None or recalls[worst] >= gate_rec else "bad")
    stage.metric("confusion matrix (rows = true)", " / ".join(" ".join(map(str, row)) for row in cm),
                 "Rows: what it really was. Columns: what the model said. Everything off the diagonal is a mistake.")
    stage.perf("training seconds", train_s)
    stage.perf("ms per prediction", predict_ms)
    stage.perf("model size KB", model_file.stat().st_size / 1024)

    if snap_proba is not None:
        k = int(np.argmax(snap_proba))
        stage.snap("prediction", classes[k])
        stage.snap("confidence", float(snap_proba[k]))
        stage.snap("probabilities", {cls: float(p) for cls, p in zip(classes, snap_proba)})
        label = os.environ.get("SNAP_LABEL", "").strip()
        if label in classes:
            stage.snap("you said", label)
            stage.snap("correct", classes[k] == label)

    stage.gate("test accuracy", acc, min=gate_acc)
    stage.gate("worst class recall", recalls.min(), min=gate_rec)
    return exam


def plot(path, cm, classes, misses, I_test, y_test, y_pred, side, color_space, title):
    """Confusion matrix · how it learned (side(ax), or recall per class) · the test frames it got wrong."""
    from matplotlib.figure import Figure  # no pyplot: headless in CI

    n_miss = min(len(misses), 8)
    fig = Figure(figsize=(13, 4.2))
    grid = fig.add_gridspec(2, 6 if n_miss else 2, width_ratios=[2.2, 2.2] + [1] * (4 if n_miss else 0))
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)), classes, rotation=30, fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("confusion matrix (test)", fontsize=10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax2 = fig.add_subplot(grid[:, 1])
    if side is not None:
        side(ax2)
    else:
        ax2.barh(classes, cm.diagonal() / np.maximum(cm.sum(axis=1), 1), color="#4caf50"); ax2.set_xlim(0, 1)
        ax2.set_title("recall per class", fontsize=10)
    for k in range(n_miss):
        i = misses[k]
        a = fig.add_subplot(grid[k // 4, 2 + k % 4])
        im = to_display(I_test[i], color_space)
        a.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255)
        a.set_title(f"{classes[y_test[i]]}\n→ {classes[y_pred[i]]}", fontsize=7, color="#b00020")
        a.axis("off")
    fig.suptitle(title + (" — misclassified test frames on the right" if n_miss else ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
