"""🔒 Shared plumbing for every sandbox: knobs, the real input frame, saving, the 👀 pictures and the results file.

A sandbox runs ONE sub-step of a stage on ONE real frame, with the stage's own recipe.py and the knobs
from its action.yaml (plus the sandbox's TRY). Nothing in here is maths: the maths lives in recipe.py.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from stages.deps import check

check()  # a friendly message instead of a traceback when this Python lacks a package

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from stages.common import ROOT, action_file, knobs_of, load_images, load_recipe, to_luma  # noqa: E402
from stages.registry import BY_KEY, ORDER  # noqa: E402

OUT = ROOT / "build" / "sandbox"   # build/ is in .gitignore: your experiments stay on your machine
LOOK_HEIGHT = 384                  # 👀 pictures are enlarged to about this height, with sharp pixels
LIGHTS = {"good": "🟢", "check": "🟠", "bad": "🔴", "info": "ℹ️"}


# ---------------------------------------------------------------- recipe & knobs

def recipe(key: str):
    """The stage's recipe.py: the same functions CI runs on every frame."""
    try:
        return load_recipe(key)
    except ModuleNotFoundError as e:
        if (e.name or "").startswith(("tensorflow", "keras")):
            raise SystemExit(f"👾 This sandbox trains a neural network, and this Python has no TensorFlow yet.\n"
                             f'       "{sys.executable}" -m pip install -r "{ROOT / "requirements-cnn.txt"}"')
        raise


def knobs(key: str, try_: dict) -> tuple[dict, dict]:
    """→ (knobs this run uses, knobs in action.yaml). TRY wins; a typo stops the run with the list of real knobs."""
    defaults = knobs_of(key)
    unknown = sorted(set(try_) - set(defaults))
    if unknown:
        raise SystemExit(f"👾 {rel(action_file(key))} has no knob {', '.join(unknown)}.\n"
                         f"   The knobs are: {', '.join(k for k in defaults if not k.startswith('gate_'))}")
    return {**defaults, **try_}, defaults


# ---------------------------------------------------------------- input

def arg() -> str | None:
    """The first command-line argument, if any (PyCharm: Run → Edit Configurations → Script parameters)."""
    return sys.argv[1].strip() if len(sys.argv) > 1 and sys.argv[1].strip() else None


def frame(previous: str, which: str = "snap") -> tuple[np.ndarray, str, Path]:
    """One real frame, as stage `previous` passed it on → (image, colour space, file).

    which: "snap" (your Step 0 photo), a class name like "drop" (its first test frame), or a file name in
    build/<previous>/images/. A path on the command line wins; that photo is used as it is.
    The pipeline runs up to `previous` first when its output is missing or older than your knobs and code.
    """
    which = arg() or which
    if Path(which).expanduser().is_file():
        img = read(Path(which).expanduser())
        h, w = img.shape[:2]
        if max(h, w) > 800:                                         # e.g. a 4000-px phone photo → 800 px
            s = 800 / max(h, w)
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        print(f"👾 Using {which} as it is: it skipped stages 1–{BY_KEY[previous].n}, so sizes and areas differ from CI.\n"
              f"   To send it through properly: python run_pipeline.py --snap {which} --to {previous}")
        return img, "bgr", Path(which).expanduser()

    refresh(previous)
    src = load_images(previous)
    d = ROOT / "build" / previous / "images"
    if which == "snap" and src.snap_idx is not None:
        i = src.snap_idx
    else:
        hits = [k for k, r in enumerate(src.rows) if r["split"] == "test" and r["label"] == which] or \
               [k for k, r in enumerate(src.rows) if r["file"] == which]
        if not hits:
            classes = ", ".join(f'"{c}"' for c in src.classes)
            raise SystemExit(f'👾 I can\'t find a frame "{which}". Use "snap", a class ({classes}), '
                             f"a file name in {rel(d)}, or a path to a photo.")
        i = hits[0]
    return src.images[i], src.color_space, d / src.rows[i]["file"]


def refresh(previous: str) -> None:
    """Run the pipeline up to `previous` when its output is missing, or older than a knob or stage you changed."""
    manifest = ROOT / "build" / previous / "metrics.json"     # every stage writes it last
    upto = ORDER[:ORDER.index(previous) + 1]
    sources = [action_file(k) for k in upto] + [ROOT / "stages" / "common.py"]
    sources += [p for k in upto for p in (ROOT / BY_KEY[k].action_path / "recipe.py",) if p.exists()]
    sources += [p for p in (ROOT / "stages").glob("s[0-9]_*.py") if int(p.name[1]) <= BY_KEY[previous].n]
    sources += [p for p in (ROOT / "snaps").iterdir() if p.is_file()] if (ROOT / "snaps").is_dir() else []
    if manifest.exists() and manifest.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
        return
    why = "isn't there yet" if not manifest.exists() else "is older than your latest change"
    print(f"👾 build/{previous}/ {why}, so I'm running the pipeline up to {BY_KEY[previous].title} first…")
    from run_pipeline import run
    with contextlib.redirect_stdout(io.StringIO()):
        run(None, previous, keep_going=True)
    if not manifest.exists():
        raise SystemExit(f"👾 The pipeline couldn't get to {BY_KEY[previous].title}. Run: python run_pipeline.py --to {previous}")


def features() -> SimpleNamespace:
    """Stage 5's output, as stage 6 receives it: feature vectors, labels, frames, and what every feature means."""
    refresh("extracting")
    from stages.s5_extracting import feature_names
    d = np.load(ROOT / "build" / "extracting" / "features.npz", allow_pickle=False)
    meta = json.loads((ROOT / "build" / "extracting" / "meta.json").read_text())
    data = SimpleNamespace(**{k: d[k] for k in d.files if k != "classes"}, classes=[str(c) for c in d["classes"]],
                           space=meta.get("color_space", "gray"), seed=knobs_of("digital_data").get("seed", 42))
    data.names = feature_names(knobs_of("extracting"), data.I_test.shape[1:])
    if len(data.names) != data.X_test.shape[1]:                # shouldn't happen; never let it crash a sandbox
        data.names = [(f"feature {i}", None) for i in range(data.X_test.shape[1])]
    return data


def pick(data: SimpleNamespace, which: str = "snap") -> tuple[np.ndarray, np.ndarray, int | None, str]:
    """One frame from features() → (feature vector, frame, true class index or None, caption).

    which: "snap" (your Step 0 photo) or a class name ("drop": its first test frame).
    """
    which = arg() or which
    if which == "snap" and len(data.X_snap):
        return data.X_snap[0], data.I_snap[0], None, "your snap"
    if which in data.classes:
        i = int(np.flatnonzero(data.y_test == data.classes.index(which))[0])
        return data.X_test[i], data.I_test[i], int(data.y_test[i]), f"test frame {i} ({which})"
    raise SystemExit(f'👾 I can\'t find a frame "{which}". Use "snap" or a class: '
                     f"{', '.join(repr(c) for c in data.classes)}.")


def new_figure(width: float, height: float):
    """A matplotlib Figure that draws off-screen (no window pops up, and it works without a display)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    return fig


def figure(fig) -> np.ndarray:
    """A matplotlib Figure as a BGR picture, for look()."""
    fig.canvas.draw()
    return cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)


def pick_input(default: Path, hint: str = "") -> Path:
    """The path on the command line, else the previous sub-step's output."""
    path = Path(arg()).expanduser() if arg() else default
    if not path.exists():
        raise SystemExit(f"👾 I need {rel(path)} first. {hint}")
    return path


def read(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    if not Path(path).is_file():
        raise SystemExit(f"👾 There's no file at {path}. Check the path, or run the previous sandbox first.")
    img = cv2.imread(str(path), flags)
    if img is None:
        raise SystemExit(f"👾 I can't read {path}. Is it an image, and is the path right?")
    return img


def read_mask(path: Path) -> np.ndarray:
    """Masks are black (0) and white (255). Re-binarise, in case an editor or a JPEG blurred them."""
    return np.where(read(path, cv2.IMREAD_GRAYSCALE) > 127, 255, 0).astype(np.uint8)


def read_photo(name: str, shape: tuple) -> np.ndarray | None:
    """The grey photo the first sub-step saved, for overlays. None if it's missing or a different size."""
    path = OUT / name
    photo = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None
    return photo if photo is not None and photo.shape == shape[:2] else None


# ---------------------------------------------------------------- output

def save(img: np.ndarray, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    cv2.imwrite(str(path), img)
    return path


def zoom(shape: tuple) -> int:
    """How many screen pixels per image pixel in the 👀 pictures."""
    return max(1, LOOK_HEIGHT // max(1, shape[0]))


def bigger(img: np.ndarray, color_space: str = "gray") -> np.ndarray:
    """BGR, enlarged with sharp pixels, so you see exactly what the pipeline sees."""
    img = to_luma(img, color_space) if color_space not in ("gray", "bgr") else img
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    s = zoom(img.shape)
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST) if s > 1 else img


def look(name: str, panels: list[tuple[str, np.ndarray]], cols: int | None = None) -> Path:
    """Side-by-side picture with a caption over each panel, `cols` per row. Same file name every run: keep it open."""
    if cols and len(panels) > cols:
        rows = [row(panels[i:i + cols]) for i in range(0, len(panels), cols)]
        width = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 255, np.uint8)]) for r in rows]
        return save(np.vstack([x for r in rows for x in (r, np.full((8, width, 3), 255, np.uint8))][:-1]), name)
    return save(row(panels), name)


def row(panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    tiles = []
    for caption, img in panels:
        img = bigger(img) if img.shape[0] < LOOK_HEIGHT // 2 else img
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        head = np.full((30, img.shape[1], 3), 255, np.uint8)
        cv2.putText(head, caption, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(np.vstack([head, img]))
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 255, np.uint8)]) for t in tiles]
    gap = np.full((h, 8, 3), 255, np.uint8)
    return np.hstack([x for t in tiles for x in (t, gap)][:-1])


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def yaml_value(v) -> str:
    """How a value is written in action.yaml: always a quoted string."""
    if isinstance(v, bool):
        return f'"{str(v).lower()}"'
    if v is None:
        return '"null"'
    if isinstance(v, (list, tuple)):
        return f'"[{", ".join(map(str, v))}]"'
    return f'"{v}"'


def write_results(key: str, step: str, title: str, story: str, files: list, used: dict, defaults: dict,
                  names: list, metrics: list, tips: list, extra: str = "") -> Path:
    """build/sandbox/<unix time>_results_stage_<step>.md: the unix time sorts your runs oldest to newest."""
    now = time.time()
    changed = [k for k in names if used[k] != defaults[k]]
    action = rel(action_file(key))
    lines = [f"# {title} · results", "",
             f"🕒 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))} · unix `{int(now)}`", "",
             "## 👾 What the Robot did", "", story, "",
             "## 📂 Files", "", "| | |", "|---|---|", *[f"| {k} | `{v}` |" for k, v in files], "",
             "## 🎛️ Knobs used", "", f"| Knob | `{action}` | This run |", "|---|---|---|",
             *[f"| `{k}` | `{defaults[k]}` | `{used[k]}`{' ✏️ TRY' if k in changed else ''} |" for k in names], ""]
    if changed:
        lines += ["## ✍️ Keep it?", "",
                  f"This run used `TRY` values that CI doesn't use yet. Better? Copy them by hand into the 🎛️ TINKER ZONE "
                  f"of `{action}`, then ▶ `recipe.py` to see what they do to every frame and to the test accuracy:", "",
                  "```yaml", *[f"  {k}:\n    default: {yaml_value(used[k])}" for k in changed], "```", ""]
    lines += ["## 📊 Metrics", "", "| | Metric | Value | | What it tells you |", "|---|---|---|---|---|",
              *[f"| {e} | {m} | **{v}** | {LIGHTS[lvl]} | {why} |" for e, m, v, lvl, why in metrics], "",
              "🟢 looks right · 🟠 worth a look · 🔴 something is off · ℹ️ for your information", "",
              *([extra, ""] if extra else []),
              "## 💡 Try next", "", *[f"- {t}" for t in tips], ""]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{int(now)}_results_stage_{step}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def report(metrics: list, outputs: list, log: Path, used: dict, defaults: dict, names: list) -> None:
    print()
    for k in names:
        if used[k] != defaults[k]:
            print(f"  ✏️ TRY {k} = {used[k]!r}   (action.yaml: {defaults[k]!r})")
    for e, m, v, lvl, _ in metrics:
        print(f"  {LIGHTS[lvl]} {e} {m}: {v}")
    for path in outputs:
        print(f"  📤 {rel(path)}")
    print(f"  📝 {rel(log)}\n")


def timed(fn, *args):
    """→ (fn's result, milliseconds it took)."""
    t0 = time.perf_counter()
    out = fn(*args)
    return out, 1000 * (time.perf_counter() - t0)
