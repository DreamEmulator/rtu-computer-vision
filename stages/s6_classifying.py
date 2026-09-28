"""Stage 6 · Classifying — running, stopped, or running dry?

🌳 Image Classification with Random Forests (Tue 06.10): 05_random-forests/classifying_random-forest/action.yaml
🧠 Image Classification with Neural Networks (Thu 08.10): 06_neural-networks/classifying_cnn/action.yaml

    python -m stages.s6_classifying random_forest
    python -m stages.s6_classifying neural_network
"""
from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from stages.common import StageReport, fresh_stage_dir, load_config, run_stage, stage_dir, to_display


def build_model(c: dict, seed: int):
    name = c.get("model", "random_forest")
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=c.get("n_estimators", 200), max_depth=c.get("max_depth"),
                                      min_samples_leaf=c.get("min_samples_leaf", 1), class_weight=c.get("class_weight"),
                                      random_state=seed, n_jobs=-1)
    if name == "svm":
        return SVC(C=c.get("svm_c", 1.0), kernel=c.get("svm_kernel", "rbf"), gamma=c.get("svm_gamma", "scale"),
                   probability=True, random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=c.get("knn_neighbors", 5))
    raise SystemExit(f"❌ Unknown model '{name}'. Choose random_forest | svm | knn")


def train_cnn(I_train, y_train, n_classes: int, c: dict, seed: int):
    """A small CNN. It learns its own features straight from the stage 4 pixels — stage 5's vectors are ignored."""
    try:
        import tensorflow as tf
    except ImportError:
        raise SystemExit("❌ The neural network needs TensorFlow → pip install -r requirements-cnn.txt")
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers
    model = tf.keras.Sequential([layers.Input(shape=I_train.shape[1:]), layers.Rescaling(1 / 255.0)])
    if (c.get("input_downscale") or 1) > 1:
        model.add(layers.AveragePooling2D(c["input_downscale"]))
    for i in range(c.get("conv_layers", 3)):
        model.add(layers.Conv2D(c.get("filters", 16) * 2 ** i, 3, padding="same", use_bias=not c.get("batch_norm")))
        if c.get("batch_norm"):
            model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D())
    model.add(layers.GlobalAveragePooling2D() if c.get("head", "flatten") == "gap" else layers.Flatten())
    model.add(layers.Dense(c.get("dense_units", 64), activation="relu"))
    model.add(layers.Dropout(c.get("dropout", 0.3)))
    model.add(layers.Dense(n_classes, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(c.get("learning_rate", 1e-3)),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    order = np.random.default_rng(seed).permutation(len(y_train))
    history = model.fit(I_train[order], y_train[order], validation_split=0.15, epochs=c.get("epochs", 40),
                        batch_size=c.get("batch_size", 32), verbose=2)
    return model, history.history


def plot_results(path, cm, classes, misses, I_test, y_test, y_pred, history, color_space, title):
    from matplotlib.figure import Figure  # no pyplot: headless in CI and doesn't disturb notebooks

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
    if history is not None:
        ax2.plot(history["accuracy"], label="train"); ax2.plot(history["val_accuracy"], label="validation")
        ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.legend(fontsize=8); ax2.set_title("learning curve", fontsize=10)
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


def main(key: str) -> None:
    cfg = load_config()
    c = cfg[key]
    seed = cfg["digital_data"].get("seed", 42)
    report = StageReport(key, cfg)
    out_dir = fresh_stage_dir(key)

    if key == "neural_network" and not c.get("enabled"):
        report.status_override = "off"
        report.note("Switched off until Thu 08.10: set enabled to \"true\" in 06_neural-networks/classifying_cnn/action.yaml")
        report.finish()
        return

    feats = stage_dir("extracting") / "features.npz"
    if not feats.exists():
        raise SystemExit("❌ No features from Extracting. Locally: python run_pipeline.py --to extracting")
    d = np.load(feats, allow_pickle=False)
    meta = json.loads((stage_dir("extracting") / "meta.json").read_text())
    classes = [str(k) for k in d["classes"]]
    y_train, y_test = d["y_train"], d["y_test"]

    t0 = time.time()
    history = None
    if key == "neural_network":
        I_train, I_test, I_snap = d["I_train"], d["I_test"], d["I_snap"]
        if I_train.ndim == 3:
            I_train, I_test, I_snap = I_train[..., None], I_test[..., None], I_snap[..., None]
        model, history = train_cnn(I_train, y_train, len(classes), c, seed)
        train_s = time.time() - t0
        t1 = time.perf_counter()
        y_pred = model.predict(I_test, verbose=0).argmax(axis=1)
        predict_ms = 1000 * (time.perf_counter() - t1) / len(y_test)
        snap_proba = model.predict(I_snap, verbose=0)[0] if len(I_snap) else None
        model.save(out_dir / "model.keras")
        report.metric("final validation accuracy", history["val_accuracy"][-1])
        report.metric("parameters", int(model.count_params()))
        title = "6 · Neural Network"
    else:
        X_train, X_test, X_snap = d["X_train"], d["X_test"], d["X_snap"]
        model = build_model(c, seed)
        if (folds := c.get("cv_folds")) and folds > 1:
            # Augmented copies share a group with their original, so they never leak across folds.
            cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            scores = cross_val_score(model, X_train, y_train, groups=d["g_train"], cv=cv)
            report.metric("cv accuracy (mean ± std)", f"{scores.mean():.3f} ± {scores.std():.3f}")
        model.fit(X_train, y_train)
        train_s = time.time() - t0
        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_ms = 1000 * (time.perf_counter() - t1) / len(y_test)
        snap_proba = model.predict_proba(X_snap)[0] if len(X_snap) else None
        joblib.dump(model, out_dir / "model.joblib", compress=3)
        title = f"6 · {c.get('model', 'random_forest').replace('_', ' ').title()}"

    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    acc = accuracy_score(y_test, y_pred)
    recalls = recall_score(y_test, y_pred, average=None, labels=range(len(classes)), zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
    (out_dir / "labels.json").write_text(json.dumps(classes))
    plot_results(out_dir / "preview.png", cm, classes, np.flatnonzero(y_pred != y_test), d["I_test"], y_test, y_pred,
                 history, meta.get("color_space", "gray"), title)

    report.metric("model", c.get("model", "cnn") if key == "random_forest" else "cnn")
    report.metric("baseline (always guess the majority)", baseline.score(np.zeros((len(y_test), 1)), y_test))
    report.metric("test accuracy", acc)
    report.metric("precision (macro)", precision_score(y_test, y_pred, average="macro", zero_division=0))
    report.metric("recall (macro)", recall_score(y_test, y_pred, average="macro", zero_division=0))
    report.metric("f1 (macro)", f1_score(y_test, y_pred, average="macro", zero_division=0))
    report.metric("recall per class", {k: float(r) for k, r in zip(classes, recalls)})
    report.metric("confusion matrix (rows = true)", " / ".join(" ".join(map(str, row)) for row in cm))
    report.perf("training seconds", train_s)
    report.perf("ms per prediction", predict_ms)
    model_file = out_dir / ("model.keras" if key == "neural_network" else "model.joblib")
    report.perf("model size KB", model_file.stat().st_size / 1024)

    if snap_proba is not None:
        k = int(np.argmax(snap_proba))
        report.snap("prediction", classes[k])
        report.snap("confidence", float(snap_proba[k]))
        report.snap("probabilities", {cls: float(p) for cls, p in zip(classes, snap_proba)})
        label = os.environ.get("SNAP_LABEL", "").strip()
        if label in classes:
            report.snap("you said", label)
            report.snap("correct", classes[k] == label)

    report.gate("test accuracy", acc, min=c.get("gate_min_accuracy"))
    report.gate("worst class recall", recalls.min(), min=c.get("gate_min_class_recall"))
    report.finish()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "random_forest"
    run_stage(which, lambda: main(which))
