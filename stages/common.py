"""Shared plumbing for every stage: knobs, stage I/O, the snap, previews, gates and reporting.

You rarely need to edit this file. The interesting code lives in every stage folder's Python file, and
the interesting *decisions* in the values of every stage folder's action.yaml.
"""
from __future__ import annotations

import base64
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from stages import webhook
from stages.registry import BY_KEY, STAGES, Stage

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
SNAP = "snap"  # split name of the photo from Step 0; it travels through every stage but never trains or tests


# ---------------------------------------------------------------- knobs

def action_file(key: str) -> Path:
    d = ROOT / BY_KEY[key].action_path
    for name in ("action.yaml", "action.yml"):
        if (d / name).exists():
            return d / name
    raise SystemExit(f"❌ No action.yaml in {d}")


def parse_value(v):
    """Action inputs are always strings: '3' → 3, '[128, 128]' → [128, 128], 'null' → None, 'true' → True."""
    if v is None or not isinstance(v, str):
        return v
    try:
        return yaml.safe_load(v) if v.strip() else None
    except yaml.YAMLError:
        return v


def knobs_of(key: str) -> dict:
    spec = yaml.safe_load(action_file(key).read_text(encoding="utf-8")) or {}
    return {name: parse_value(str(i.get("default", ""))) for name, i in (spec.get("inputs") or {}).items()}


def load_config() -> dict:
    """{stage key: {knob: value}} from the defaults in every action.yaml.

    In CI each action also passes its actual inputs as CV_KNOBS (JSON) for its own stage, which wins,
    so a value given with `with:` in pipeline.yml is what the stage really uses.
    """
    cfg = {s.key: knobs_of(s.key) for s in STAGES}
    stage, knobs = os.environ.get("CV_STAGE"), os.environ.get("CV_KNOBS")
    if stage in cfg and knobs:
        cfg[stage].update({k: parse_value(v) for k, v in json.loads(knobs).items()})
    return cfg


def stage_dir(key: str) -> Path:
    return BUILD / key


def fresh_stage_dir(key: str) -> Path:
    d = stage_dir(key)
    if d.exists():
        shutil.rmtree(d)
    (d / "images").mkdir(parents=True)
    return d


def odd(k: int) -> int:
    """Most OpenCV kernels need an odd size so they have a centre pixel."""
    k = max(1, int(k))
    return k if k % 2 else k + 1


# ---------------------------------------------------------------- image sets

@dataclass
class ImageSet:
    rows: list[dict]              # manifest: file, label, split (train | test | snap), source
    images: list[np.ndarray]
    meta: dict = field(default_factory=dict)

    @property
    def color_space(self) -> str:
        return self.meta.get("color_space", "gray")

    @property
    def classes(self) -> list[str]:
        return self.meta.get("classes") or sorted({r["label"] for r in self.rows if r["split"] != SNAP})

    @property
    def data_idx(self) -> list[int]:
        return [i for i, r in enumerate(self.rows) if r["split"] != SNAP]

    @property
    def snap_idx(self) -> int | None:
        return next((i for i, r in enumerate(self.rows) if r["split"] == SNAP), None)

    def data(self, images: list | None = None) -> list:
        images = self.images if images is None else images
        return [images[i] for i in self.data_idx]


def load_images(key: str) -> ImageSet:
    """Load the output of a stage (what the next stage receives)."""
    d = stage_dir(key)
    manifest = d / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"❌ No output of stage '{key}' in {d}.\n"
                         f"   Locally: python run_pipeline.py --to {key}    In CI: did the previous job succeed?")
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    images = [cv2.imread(str(d / "images" / r["file"]), cv2.IMREAD_UNCHANGED) for r in rows]
    meta_file = d / "meta.json"
    return ImageSet(rows, images, json.loads(meta_file.read_text()) if meta_file.exists() else {})


def save_images(key: str, iset: ImageSet) -> None:
    """PNG is lossless, so no stage silently degrades the next one."""
    d = stage_dir(key)
    (d / "images").mkdir(parents=True, exist_ok=True)
    for r, img in zip(iset.rows, iset.images):
        cv2.imwrite(str(d / "images" / r["file"]), img)
    with open(d / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(iset.rows[0].keys()))
        w.writeheader()
        w.writerows(iset.rows)
    (d / "meta.json").write_text(json.dumps(iset.meta, indent=2))


# ---------------------------------------------------------------- colour helpers
LUMA = {"gray": None, "bgr": None, "hsv": 2, "lab": 0, "ycrcb": 0}   # channel that carries brightness
TO_RGB = {"bgr": cv2.COLOR_BGR2RGB, "hsv": cv2.COLOR_HSV2RGB, "lab": cv2.COLOR_LAB2RGB, "ycrcb": cv2.COLOR_YCrCb2RGB}


def to_luma(img: np.ndarray, color_space: str) -> np.ndarray:
    """Single-channel brightness image, whatever colour space we are in."""
    if img.ndim == 2:
        return img
    if color_space == "bgr":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(img[..., LUMA.get(color_space, 0)])


def on_luma(img: np.ndarray, color_space: str, fn) -> np.ndarray:
    """Apply fn to the brightness channel only, leaving colour information intact."""
    if img.ndim == 2:
        return fn(img)
    if color_space == "bgr":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = fn(np.ascontiguousarray(lab[..., 0]))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out = img.copy()
    c = LUMA.get(color_space, 0)
    out[..., c] = fn(np.ascontiguousarray(img[..., c]))
    return out


def to_display(img: np.ndarray, color_space: str = "bgr") -> np.ndarray:
    """RGB (or gray) array that matplotlib shows correctly."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, TO_RGB.get(color_space, cv2.COLOR_BGR2RGB))


def to_bgr(img: np.ndarray, color_space: str = "gray") -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(to_display(img, color_space), cv2.COLOR_RGB2BGR)


def data_uri(img: np.ndarray, color_space: str = "gray", min_side: int = 256) -> str:
    """Small PNG as a data: URI, upscaled with sharp pixels so the page shows what the pipeline sees."""
    img = to_bgr(img, color_space)
    scale = max(1, int(np.ceil(min_side / max(1, min(img.shape[:2])))))
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def timed(fn, items) -> tuple[list, float]:
    """Apply fn to every item; return results and milliseconds per item."""
    t0 = time.perf_counter()
    out = [fn(x) for x in items]
    return out, 1000 * (time.perf_counter() - t0) / max(1, len(items))


# ---------------------------------------------------------------- previews

def sample_indices(rows: list[dict], per_class: int = 3) -> list[int]:
    """The same test images in every stage, so you can follow one frame through the pipeline."""
    picked = []
    for c in sorted({r["label"] for r in rows if r["split"] != SNAP}):
        picked += [i for i, r in enumerate(rows) if r["label"] == c and r["split"] == "test"][:per_class]
    return picked


def contact_sheet(path: Path, rows: list[dict], strips: list[tuple[str, list[np.ndarray], str]],
                  title: str, per_class: int = 3) -> None:
    """strips = [(row label, images aligned with rows, colour space), ...]. The snap gets its own column."""
    from matplotlib.figure import Figure  # no pyplot: headless in CI

    idx = sample_indices(rows, per_class)
    snap = next((i for i, r in enumerate(rows) if r["split"] == SNAP), None)
    if snap is not None:
        idx.append(snap)
    fig = Figure(figsize=(1.7 * len(idx), 1.9 * len(strips) + 0.5))
    axes = fig.subplots(len(strips), len(idx), squeeze=False)
    for r, (name, imgs, space) in enumerate(strips):
        for c, i in enumerate(idx):
            ax = axes[r][c]
            im = to_display(imgs[i], space)
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("Your snap" if i == snap else rows[i]["label"], fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


# ---------------------------------------------------------------- report, gates & webhooks

def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{k} {_fmt(x)}" for k, x in v.items())
    return str(v)


def _plain(v):
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


class StageReport:
    """Collects metrics, gates and the snap's journey; writes metrics.json, the CI summary and the webhook.

    A gate is a promise the startup makes ("we never ship below 85 % accuracy"). If one fails, the stage
    exits with code 1: the job turns red and the stages after it don't run, like a failing unit test.
    """

    def __init__(self, key: str, cfg: dict):
        self.key, self.cfg, self.stage = key, cfg, BY_KEY[key]
        self.t0 = time.time()
        self.metrics: dict = {}
        self.gates: list[dict] = []
        self.notes: list[str] = []
        self.performance: dict = {}
        self.snap_metrics: dict = {}
        self.snap_images: dict = {}
        self.status_override: str | None = None
        self.why: dict = {}            # metric → (light, one sentence on what it tells you)
        self.tips: list[str] = []
        self.since: list[str] = []     # what changed since your last run (filled in by finish)

    def metric(self, name: str, value, why: str | None = None, level: str | None = None):
        """level: good | check | bad | info → 🟢 🟠 🔴 ℹ️ in the report. why: what the number tells you."""
        self.metrics[name] = _plain(value)
        if why or level:
            self.why[name] = (level or "info", why or "")
        return self.metrics[name]

    def tip(self, text: str):
        """💡 Something to try next, shown at the end of the report."""
        self.tips.append(text)

    def perf(self, name: str, value):
        self.performance[name] = _plain(value)

    def note(self, text: str):
        self.notes.append(text)
        print(f"  • {text}")

    def snap(self, name: str, value):
        self.snap_metrics[name] = _plain(value)

    def snap_image(self, name: str, img: np.ndarray, color_space: str = "gray"):
        d = stage_dir(self.key) / "snap"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{name}.png"), to_bgr(img, color_space))
        self.snap_images[name] = data_uri(img, color_space)

    def gate(self, name: str, value, *, min=None, max=None):
        """Skipped when the threshold is null in the GATE ZONE."""
        if min is None and max is None:
            return True
        value = float(value)
        ok = (min is None or value >= min) and (max is None or value <= max)
        self.gates.append({"name": name, "value": round(value, 4),
                           "rule": f"≥ {min}" if min is not None else f"≤ {max}", "ok": ok})
        return ok

    @property
    def passed(self) -> bool:
        return all(g["ok"] for g in self.gates)

    @property
    def status(self) -> str:
        return self.status_override or ("passed" if self.passed else "failed")

    def _links(self) -> str:
        s = self.stage
        repo = os.environ.get("GITHUB_REPOSITORY")
        ref = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME") or "main"
        gh = f"https://github.com/{repo}/blob/{ref}" if repo else ""
        files = [("📖", "README", "README.md"), ("🎛️", "Values", "action.yaml"), ("🐍", "Code", s.code)]
        return " · ".join(f"{e} [{t}]({gh}/{s.folder}/{f})" if repo else f"{e} `{s.folder}/{f}`" for e, t, f in files)

    def markdown(self) -> str:
        """Most important first: may it pass on, what changed, the numbers, what to try. The full values last."""
        s = self.stage
        out = [f"## {s.emoji} {s.n} · {s.title}  ·  📅 {s.lecture}", "", f"> *{s.question}*", ""]
        if self.status == "off":
            out += ["⏸️ Switched off: set `enabled` to `\"true\"` in the values (action.yaml).", ""]
        if self.gates:
            verdict = "✅ Ready to pass on" if self.passed else "❌ Not ready to pass on"
            out += [f"**🚦 {verdict}:** " + " · ".join(f"{'✅' if g['ok'] else '❌'} {g['name']} {_fmt(g['value'])} (must be {g['rule']})"
                                                       for g in self.gates), ""]
        if self.since:
            out += ["**🔁 Since your last run:** " + " · ".join(self.since), ""]
        if self.metrics and self.why:
            out += ["| | Metric | Value | What it tells you |", "|---|---|---|---|"]
            out += [f"| {LIGHTS[self.why[k][0]] if k in self.why else ''} | {k} | {_fmt(v)} | {self.why[k][1] if k in self.why else ''} |"
                    for k, v in self.metrics.items()] + ["", "🟢 looks right · 🟠 worth a look · 🔴 something is off · ℹ️ for your information", ""]
        elif self.metrics:
            out += ["| Metric | Value |", "|---|---|"] + [f"| {k} | {_fmt(v)} |" for k, v in self.metrics.items()] + [""]
        if self.snap_metrics:
            out += ["**📸 Your snap** " + " · ".join(f"{k}: {_fmt(v)}" for k, v in self.snap_metrics.items()), ""]
        if self.tips:
            out += ["**💡 What to try next**", "", *[f"- {t}" for t in self.tips], ""]
        if self.performance:
            out += ["**⚡ Performance** " + " · ".join(f"{k}: {_fmt(v)}" for k, v in self.performance.items()), ""]
        out += [f"- {t}" for t in self.notes] + ([""] if self.notes else [])
        knobs = {k: v for k, v in self.cfg.get(self.key, {}).items() if not k.startswith("gate_")}
        if knobs:
            out += [f"<details><summary>🎛️ All {len(knobs)} values this run used (<code>{s.action_path}/action.yaml</code>)</summary>", "",
                    "| Value | |", "|---|---|", *[f"| `{k}` | `{_fmt(v)}` |" for k, v in knobs.items()], "", "</details>", ""]
        out.append(self._links())
        return "\n".join(out) + "\n"

    def finish(self) -> None:
        duration = round(time.time() - self.t0, 2)
        d = stage_dir(self.key)
        d.mkdir(parents=True, exist_ok=True)
        record = {"stage": self.key, "n": self.stage.n, "title": self.stage.title, "status": self.status,
                  "passed": self.status in ("passed", "off"), "duration_s": duration,
                  "knobs": _plain(self.cfg.get(self.key, {})), "metrics": self.metrics, "performance": self.performance,
                  "gates": self.gates, "notes": self.notes, "snap": self.snap_metrics or None}
        (d / "metrics.json").write_text(json.dumps(record, indent=2, default=str))
        self.since = self._compare_with_last_run(record)
        (d / "report.md").write_text(self.markdown(), encoding="utf-8")

        print(f"\n{self.stage.emoji}  {self.stage.n} · {self.stage.title}   [{self.status}, {duration}s]")
        for k, v in {**self.metrics, **self.performance}.items():
            print(f"   {k:<34} {_fmt(v)}")
        for k, v in self.snap_metrics.items():
            print(f"   📸 {k:<31} {_fmt(v)}")
        for g in self.gates:
            print(f"   {'✅' if g['ok'] else '❌'} gate {g['name']}: {_fmt(g['value'])} (must be {g['rule']})")

        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
                f.write(self.markdown())

        webhook.emit("stage.finished", {
            "stage": webhook.stage_info(self.key), "status": self.status, "duration_s": duration,
            "knobs": record["knobs"], "metrics": self.metrics, "performance": self.performance, "gates": self.gates,
            "snap": {"metrics": self.snap_metrics, "images": self.snap_images} if (self.snap_metrics or self.snap_images) else None,
            "error": None})

        for g in self.gates:
            if not g["ok"]:
                print(f"::error title={self.stage.title} gate failed: {g['name']}::{_fmt(g['value'])} is not {g['rule']}"
                      f" — change a value in {self.stage.action_path}/action.yaml")
        if self.status == "failed":
            sys.exit(1)


    def _compare_with_last_run(self, record: dict) -> list[str]:
        """Keep every run's values and numbers in build/history/<stage>/, and say what changed since the last one."""
        history = BUILD / "history" / self.key
        history.mkdir(parents=True, exist_ok=True)
        runs = sorted(history.glob("*.json"))
        (history / f"{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(record, indent=2, default=str))
        if not runs:
            return []
        last = json.loads(runs[-1].read_text())
        changes = [f"`{k}` {_fmt(last['knobs'].get(k))} → **{_fmt(v)}**" for k, v in record["knobs"].items()
                   if not k.startswith("gate_") and last["knobs"].get(k) != v]
        for k, v in record["metrics"].items():
            old = last["metrics"].get(k)
            if isinstance(v, (int, float)) and isinstance(old, (int, float)) and not isinstance(v, bool) and old != v:
                changes.append(f"{k} {_fmt(old)} → **{_fmt(v)}** ({'+' if v > old else ''}{_fmt(v - old)})")
        return changes or ["the same values and the same numbers as last time"]


LIGHTS = {"good": "🟢", "check": "🟠", "bad": "🔴", "info": "ℹ️"}


def run_stage(key: str, main) -> None:
    """Entry point for every stage: runs main(), and reports crashes to the log, error.json and the webhook."""
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        d = stage_dir(key)
        d.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        (d / "error.json").write_text(json.dumps({"stage": key, "message": str(e), "traceback": tb}, indent=2))
        webhook.emit("stage.finished", {"stage": webhook.stage_info(key), "status": "error",
                                        "error": {"message": f"{type(e).__name__}: {e}", "traceback": tb[-4000:]}})
        print(f"::error title={BY_KEY[key].title} crashed::{type(e).__name__}: {e}")
        raise
