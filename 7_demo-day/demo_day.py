"""
🎤 Stage 7 · Demo Day — what do we show the investors (and the hospital)?
════════════════════════════════════════════════════════════════════════
Lecture: Course Summary · Tue 13.10 · Why this stage exists: README.md · The values: action.yaml

This is the file CI runs last, even when a gate failed earlier, so the report always shows where the pipeline
stopped. Press ▶ in PyCharm (or `python 7_demo-day/demo_day.py`) after running the pipeline:

    build/demo_day/index.html   one page with the whole story: your snap's journey, every stage's numbers,
                                gates and before/after picture. CI publishes it on GitHub Pages.

It collects what every stage folder wrote to build/<stage>/: metrics.json and before_after.png.
"""
from __future__ import annotations

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # 🔒 finds stages/

import base64
import html
import json
import os
from datetime import datetime, timezone

from stages import webhook
from stages.common import _fmt, fresh_stage_dir, load_config, stage_dir
from stages.stage import run
from stages.registry import BY_KEY, ORDER

KEY = "demo_day"
PIPELINE = [k for k in ORDER if k != KEY]
HEADLINE = {"digital_data": "images", "cleaning": "noise σ after", "improving": "contrast of dimmest 5% after",
            "segmenting": "foreground %", "extracting": "feature vector length",
            "random_forest": "test accuracy", "neural_network": "test accuracy"}
JOURNEY = [("digital_data", "camera", "your snap"), ("digital_data", "output", "digital data"),
           ("cleaning", "output", "cleaned"), ("improving", "output", "improved"),
           ("segmenting", "overlay", "segmented"), ("extracting", "hog", "HOG features")]


def collect() -> dict[str, dict | None]:
    out = {}
    for k in PIPELINE:
        f, err = stage_dir(k) / "metrics.json", stage_dir(k) / "error.json"
        out[k] = json.loads(f.read_text()) if f.exists() else ({"status": "error", **json.loads(err.read_text())}
                                                                if err.exists() else None)
    return out


def status(m: dict | None) -> str:
    return "skipped" if m is None else m.get("status", "passed" if m.get("passed") else "failed")


def overall(results: dict) -> str:
    return "passed" if all(status(m) in ("passed", "off") for m in results.values()) else "failed"


ICON = {"passed": "✅", "failed": "❌", "skipped": "⏭️", "off": "⏸️", "error": "💥"}


def summary_markdown(results: dict, c: dict) -> str:
    lines = [f"# 🎤 Demo Day — {c.get('project_name')}", "", f"> {c.get('pitch')}", "",
             "| | Stage | Headline | Gates | ⏱ |", "|---|---|---|---|---|"]
    for k, m in results.items():
        s, st = BY_KEY[k], status(m)
        metrics = (m or {}).get("metrics", {})
        head = f"{HEADLINE[k]}: **{_fmt(metrics[HEADLINE[k]])}**" if HEADLINE[k] in metrics else st
        gates = " ".join(("✅" if g["ok"] else "❌") + " " + g["name"] for g in (m or {}).get("gates", [])) or "—"
        lines.append(f"| {ICON[st]} | {s.emoji} {s.n} · {s.title} | {head} | {gates} | {(m or {}).get('duration_s', '—')} s |")
    preds = predictions(results)
    if preds:
        lines += ["", "**📸 Your snap:** " + " · ".join(f"{BY_KEY[k].title.split('· ')[-1]} says **{p['prediction']}** "
                                                      f"({100 * p['confidence']:.0f}%)" for k, p in preds.items())]
    return "\n".join(lines) + "\n\n📦 The full report is the `demo_day` artifact (open `index.html`).\n"


def predictions(results: dict) -> dict:
    return {k: m["snap"] for k, m in results.items()
            if m and isinstance(m.get("snap"), dict) and "prediction" in m["snap"]}


def journey_html(results: dict) -> str:
    esc = html.escape
    cells = []
    for key, name, caption in JOURNEY:
        f = stage_dir(key) / "snap" / f"{name}.png"
        if f.exists():
            cells.append(f'<figure><img alt="{esc(caption)}" src="data:image/png;base64,'
                         f'{base64.b64encode(f.read_bytes()).decode()}"><figcaption>{esc(caption)}</figcaption></figure>')
    for k, p in predictions(results).items():
        cells.append(f'<figure class="verdict-card"><strong>{esc(p["prediction"])}</strong>'
                     f'<span>{100 * p["confidence"]:.0f}% sure</span><figcaption>{esc(BY_KEY[k].title.split("· ")[-1])}'
                     f'</figcaption></figure>')
    return f'<section class="journey"><h2>Your snap, stage by stage</h2><div class="strip-row">{"".join(cells)}</div></section>' if cells else ""


CSS = """
:root{--mint:#e3f6ec;--mint-2:#cfeedd;--ink:#1f4a45;--ink-2:#4d706b;--go:#7ed957;--stop:#e0584b;--idle:#c4d3cc;--paper:#fbfdfc}
*{box-sizing:border-box}
html{background:var(--mint)}
body{margin:0;color:var(--ink);font:17px/1.55 Jost,"Avenir Next",Avenir,"Segoe UI",system-ui,sans-serif;
     background:radial-gradient(120% 80% at 20% 0%,#f4fcf8 0%,var(--mint) 60%);min-height:100vh;
     padding:env(safe-area-inset-top,0) 0 env(safe-area-inset-bottom,0)}
main{max-width:1080px;margin:0 auto;padding:48px 24px 80px}
h1{font-size:clamp(2.4rem,6vw,4rem);line-height:1;margin:0;font-weight:600;text-decoration:underline;
   text-decoration-thickness:.07em;text-underline-offset:.12em}
.pitch{font-size:1.25rem;color:var(--ink-2);margin:.6rem 0 0;max-width:60ch}
.run{margin-top:1.4rem;color:var(--ink-2);font-size:.95rem}
.verdict{display:inline-block;margin-top:1.2rem;padding:.35rem .9rem;border:2px solid var(--ink);font-weight:600}
.verdict.failed{border-color:var(--stop);color:var(--stop)}
.strip{display:grid;grid-template-columns:repeat(2,1fr);gap:28px;margin:44px 0 56px}
.lane h2{font-size:1.35rem;margin:0 0 12px;font-weight:500}
.lane ol{list-style:none;margin:0;padding:0 0 0 10px;border-left:3px solid var(--ink)}
.lane li{margin:0 0 10px}
.lane a{display:block;transform:skewX(-18deg);margin-left:calc(var(--i)*22px);padding:14px 20px;
        border:2px solid var(--ink);color:var(--ink);text-decoration:none;font-style:italic;font-weight:600;
        font-size:1.15rem;width:max-content;min-width:60%}
.lane a>span{display:inline-block;transform:skewX(18deg)}
.lane a small{font-style:normal;font-weight:400;display:block;font-size:.85rem;color:inherit;opacity:.8}
.passed a{background:var(--go)} .failed a{background:var(--stop);color:#fff} .skipped a{background:var(--idle);border-style:dashed}
.lane a:focus-visible{outline:3px solid var(--ink);outline-offset:4px}
section{padding:36px 0;border-top:2px solid var(--mint-2)}
section header{display:flex;flex-wrap:wrap;align-items:baseline;gap:.4rem 1.2rem}
section h3{font-size:1.7rem;margin:0;font-weight:600}
section .when{color:var(--ink-2)}
section .q{font-size:1.15rem;font-style:italic;margin:.5rem 0 1.2rem;max-width:60ch}
.body{display:grid;grid-template-columns:minmax(0,1fr);gap:24px}
.body>div{max-width:560px}
table{border-collapse:collapse;width:100%;font-size:.95rem}
td{padding:5px 8px 5px 0;border-bottom:1px solid var(--mint-2);vertical-align:top}
td:last-child{text-align:right;font-variant-numeric:tabular-nums;font-weight:500}
.gates{margin:0 0 14px;padding:0;list-style:none}
.gates li{padding:4px 0}
.gates .ok::before{content:"✓ ";color:#2e7d32;font-weight:700}
.gates .no{color:var(--stop);font-weight:600}.gates .no::before{content:"✗ "}
figure{margin:0;overflow-x:auto}
figure img{max-width:100%;display:block;background:#fff;border:2px solid var(--ink)}
.none{color:var(--ink-2);font-style:italic}
details{margin-top:48px}
summary{cursor:pointer;font-weight:600}
pre{background:var(--paper);border:2px solid var(--mint-2);padding:16px;overflow-x:auto;font-size:.85rem}
.off a{background:var(--paper);border-style:dashed;color:var(--ink-2)} .error a{background:var(--stop);color:#fff}
.journey{border-top:none;padding-top:0}
.journey h2{font-size:1.35rem;font-weight:500;margin:0 0 14px}
.strip-row{display:flex;flex-wrap:wrap;gap:14px;padding-bottom:8px}
.strip-row figure{flex:0 0 auto;margin:0;text-align:center}
.strip-row img{width:150px;height:150px;object-fit:contain;border:2px solid var(--ink);background:#fff;image-rendering:pixelated}
.strip-row figcaption{font-size:.9rem;color:var(--ink-2);margin-top:4px}
.verdict-card{width:150px;height:150px;border:2px solid var(--ink);background:var(--go);display:flex;flex-direction:column;
  justify-content:center;align-items:center;gap:4px}
.verdict-card strong{font-size:1.3rem;font-style:italic}
.verdict-card figcaption{margin:0}
@media (max-width:760px){.strip{grid-template-columns:1fr}.lane a{min-width:0}}
"""


def build_html(results: dict, c: dict) -> str:
    esc = html.escape
    ok = overall(results) == "passed"
    repo = os.environ.get("GITHUB_REPOSITORY", "local run")
    sha = os.environ.get("GITHUB_SHA", "")[:7]
    ref = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME") or ""
    when = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")

    label = {"passed": "gates passed", "failed": "gate failed", "skipped": "did not run", "off": "switched off",
             "error": "crashed"}

    def lane(title, keys):
        items = []
        for i, k in enumerate(keys):
            s, st = BY_KEY[k], status(results[k])
            items.append(f'<li class="{st}"><a href="#{k}" style="--i:{i}"><span>{esc(s.title)}'
                         f'<small>{label[st]}</small></span></a></li>')
        return f'<div class="lane"><h2>{title}</h2><ol>{"".join(items)}</ol></div>'

    sections = []
    for k, m in results.items():
        s, st = BY_KEY[k], status(m)
        if m is None:
            body = '<p class="none">This stage did not run: an earlier stage failed its gates.</p>'
        elif st == "error":
            body = f'<p class="none">This stage crashed: {esc(m.get("message", ""))}</p>'
        elif st == "off":
            body = f'<p class="none">Switched off in {esc(s.action_path)}/action.yaml.</p>'
        else:
            gates = "".join(f'<li class="{"ok" if g["ok"] else "no"}">{esc(g["name"])}: {_fmt(g["value"])} '
                            f'(must be {esc(g["rule"])})</li>' for g in m["gates"])
            rows = "".join(f"<tr><td>{esc(k.strip())}</td><td>{esc(_fmt(v))}</td></tr>"
                           for k, v in m["metrics"].items())
            prev = stage_dir(k) / "before_after.png"
            img = (f'<figure><img alt="Preview of {esc(s.title)}" src="data:image/png;base64,'
                   f'{base64.b64encode(prev.read_bytes()).decode()}"></figure>') if prev.exists() else ""
            body = f'<div class="body"><div><ul class="gates">{gates}</ul><table>{rows}</table></div>{img}</div>'
        sections.append(f'<section id="{k}"><header><h3>{s.n}. {esc(s.title)}</h3>'
                        f'<span class="when">{esc(s.topic_title)}, lecture {esc(s.lecture)}</span></header>'
                        f'<p class="q">{esc(s.question)}</p>{body}</section>')

    knobs = "\n".join(f"{k}: " + json.dumps(m.get("knobs", {}), default=str) for k, m in results.items() if m)
    config = esc(knobs or "no stage ran")
    verdict = ("Every gate passed. Ship it." if ok
               else "Not ready to ship: at least one gate failed or a stage did not run.")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{esc(c.get('project_name'))} · Demo Day</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Jost:ital,wght@0,400;0,500;0,600;1,500;1,600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><main>
<h1>{esc(c.get('project_name'))}</h1>
<p class="pitch">{esc(c.get('pitch'))}</p>
<p class="run">{esc(repo)} {esc(ref)} {esc(sha)} — {when}</p>
<p class="verdict {'passed' if ok else 'failed'}">{verdict}</p>
<nav class="strip" aria-label="Pipeline stages">{lane("Pre-Processing", PIPELINE[:3])}{lane("Processing", PIPELINE[3:])}</nav>
{journey_html(results)}
{''.join(sections)}
<details><summary>Tinker-zone values used in this run</summary><pre>{config}</pre></details>
</main></body></html>"""


def main() -> None:
    c = load_config()[KEY]
    out = fresh_stage_dir(KEY)
    results = collect()
    (out / "index.html").write_text(build_html(results, c), encoding="utf-8")
    md = summary_markdown(results, c)
    print(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md)
    (out / "metrics.json").write_text(json.dumps({"stage": KEY, "status": overall(results)}, indent=2))
    webhook.emit("pipeline.finished", {
        "status": overall(results),
        "stages": [{"key": k, "title": BY_KEY[k].title, "status": status(m), "duration_s": (m or {}).get("duration_s")}
                   for k, m in results.items()],
        "predictions": predictions(results),
        "total_s": round(sum((m or {}).get("duration_s") or 0 for m in results.values()), 2)})
    print(f"📄 Report: {out / 'index.html'}")


if __name__ == "__main__":
    run(KEY, main)
