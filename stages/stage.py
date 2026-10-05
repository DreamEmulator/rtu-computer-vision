"""🔒 What every stage file needs that isn't the maths: its values, its input, saving, the report, the
before/after picture and the gates. A stage file (e.g. 4_segmenting/segmenting.py) then reads top to bottom:

    stage = Stage("segmenting")                  # the values from action.yaml
    frames = stage.frames()                      # what stage 3 passed on
    for i, img in enumerate(frames.images):
        out = segment(img, stage.knobs, stage.steps(i))   # the maths; steps() collects the pictures
    stage.save(outputs)
    stage.metric(...), stage.gate(...)
    stage.finish()                               # report.md, before_after.png, metrics.json, CI summary, webhook

Run a stage file on its own (▶ in PyCharm, or `python 4_segmenting/segmenting.py`) and it first runs the stages
before it, when their output is missing or older than a value or a file you changed.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path

from stages.deps import check

check()  # a friendly message instead of a traceback when this Python lacks a package

from stages.common import (BUILD, ROOT, SNAP, ImageSet, StageReport, action_file, fresh_stage_dir,  # noqa: E402
                           load_config, load_images, run_stage, save_images, to_display)
from stages.registry import BY_KEY, ORDER  # noqa: E402


class Stage:
    def __init__(self, key: str):
        self.key, self.info = key, BY_KEY[key]
        self.cfg = load_config()
        self.knobs: dict = self.cfg[key]                 # the values from action.yaml (in CI: what the action passed in)
        self.input: ImageSet | None = None
        self.dir = fresh_stage_dir(key)
        self.report = StageReport(key, self.cfg)
        self.metric, self.perf, self.tip, self.note = self.report.metric, self.report.perf, self.report.tip, self.report.note
        self.snap, self.snap_image, self.gate = self.report.snap, self.report.snap_image, self.report.gate
        self._pictured: dict[int, str] = {}
        self._steps: dict[int, list] = {}

    def frames(self) -> ImageSet:
        """What the previous stage passed on: every frame, its label and split, and your snap."""
        self.input = load_images(self.info.previous)
        self.picture_rows(self.input.rows)
        return self.input

    def picture_rows(self, rows: list[dict]) -> None:
        """Which frames go in before_after.png (frames() does this for you)."""
        self._pictured = pictured(rows)

    def steps(self, i: int) -> list | None:
        """A list to collect frame i's step pictures in, if it's one of the frames in before_after.png; else None."""
        if i not in self._pictured:
            return None
        return self._steps.setdefault(i, [])

    def save(self, images: list, meta: dict | None = None, rows: list[dict] | None = None) -> None:
        """What the next stage receives: one image per frame, PNG (lossless)."""
        rows = rows if rows is not None else self.input.rows
        save_images(self.key, ImageSet(rows, images, {**(self.input.meta if self.input else {}), **(meta or {})}))

    def finish(self, title: str) -> None:
        rows = [(label, self._steps[i]) for i, label in self._pictured.items() if self._steps.get(i)]
        if rows:
            before_after(self.dir / "before_after.png", rows, title)
        self.report.finish()


def picture(steps: list | None, name: str, image, space: str = "gray") -> None:
    """Add one step's picture to before_after.png. Does nothing for the frames that aren't in it (steps is None)."""
    if steps is not None:
        steps.append((name, image, space))


def pictured(rows: list[dict]) -> dict[int, str]:
    """The frames in before_after.png: your snap, then the first test frame of every class."""
    out = {i: "your snap" for i, r in enumerate(rows) if r["split"] == SNAP}
    for c in sorted({r["label"] for r in rows if r["split"] != SNAP}):
        i = next((i for i, r in enumerate(rows) if r["label"] == c and r["split"] == "test"), None)
        if i is not None:
            out[i] = f"{c} (test)"
    return out


def before_after(path: Path, rows: list[tuple[str, list]], title: str) -> None:
    """One row per frame, one column per step: [(row label, [(step name, image, colour space), ...]), ...]."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    cols = max(len(steps) for _, steps in rows)
    fig = Figure(figsize=(1.95 * cols, 2.05 * len(rows) + 0.5))
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(rows), cols, squeeze=False)
    for r, (label, steps) in enumerate(rows):
        for c in range(cols):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if c >= len(steps):
                ax.axis("off")
                continue
            name, img, space = steps[c]
            im = to_display(img, space)
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255)
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


# ---------------------------------------------------------------- running a stage file on its own

def code_files(key: str) -> list[Path]:
    """The files that decide a stage's output: its values and its code."""
    s = BY_KEY[key]
    return [action_file(key), ROOT / s.folder / s.code]


def ensure(key: str) -> None:
    """Make sure build/<key>/ is there and up to date: run the pipeline up to it if it's missing or older than
    a value, a code file or a snap you changed."""
    done = BUILD / key / "metrics.json"                   # every stage writes it last
    upto = ORDER[:ORDER.index(key) + 1]
    sources = [p for k in upto for p in code_files(k)] + [ROOT / "stages" / "common.py"]
    sources += [p for p in (ROOT / "snaps").iterdir() if p.is_file()] if (ROOT / "snaps").is_dir() else []
    if done.exists() and done.stat().st_mtime >= max(p.stat().st_mtime for p in sources if p.exists()):
        return
    why = "isn't there yet" if not done.exists() else "is older than your latest change"
    print(f"👾 build/{key}/ {why}, so I'm running the pipeline up to {BY_KEY[key].title} first…")
    from run_pipeline import run
    with contextlib.redirect_stdout(io.StringIO()):
        run(None, key, keep_going=True)
    if not done.exists():
        raise SystemExit(f"👾 The pipeline couldn't get to {BY_KEY[key].title}. Run: python run_pipeline.py --to {key}")


def load_code(key: str):
    """A stage folder's Python file as a module. Stage folders start with a digit, so they can't be imported by name.
    Also how one stage uses another's functions, e.g. the forest naming stage 5's features."""
    s = BY_KEY[key]
    name = f"stage_{key}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / s.folder / s.code)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(sys.modules[name])
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def run(key: str, main) -> None:
    """▶ One stage on your machine (or in CI, where the stages before it already ran in their own jobs)."""
    s = BY_KEY[key]
    if s.previous and not os.environ.get("CI"):
        ensure(s.previous)
    try:
        run_stage(key, main)
    finally:
        out = BUILD / key
        if (out / "report.md").exists():
            print(f"\n📝 The report:  {(out / 'report.md').relative_to(ROOT)}")
        if (out / "before_after.png").exists():
            print(f"👀 The picture: {(out / 'before_after.png').relative_to(ROOT)}")
        print(f"🎛️ The values:  {s.action_path}/action.yaml   (change one, press ▶ again, read 'Since your last run')")
