"""Draw the Drip Detector pipeline (after the course slide) with the week's stages highlighted.

    python -m tools.make_pipeline_svg      # writes <stage folder>/pipeline.svg for every stage
"""
from __future__ import annotations

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INK, GO, PALE, PALE_EDGE, PALE_TEXT, MINT = "#1f4a45", "#7ed957", "#f4fbf7", "#a9c4b9", "#8aa89c", "#e3f6ec"
FONT = "'Avenir Next', Avenir, Jost, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
PRE = ["Digital Data", "Cleaning", "Improving"]
PRO = ["Segmenting", "Extracting", "Classifying"]
TOPICS = {   # stage folder: (highlighted boxes, sub-label)
    "1_digital-data": (["Digital Data"], None),
    "2_cleaning": (["Cleaning"], None),
    "3_improving": (["Improving"], None),
    "4_segmenting": (["Segmenting"], None),
    "5_extracting": (["Extracting"], None),
    "6_random-forest": (["Classifying"], "random forest"),
    "6_neural-network": (["Classifying"], "CNN"),
    "7_demo-day": (PRE + PRO + ["Inference"], None),
    ".": (PRE + PRO + ["Inference"], None),          # the root README: everything, no marker
}
W, H, SKEW, SW, SH = 190, 56, 30, 38, 62


def shape(x0, y0, title, on, sub=None, sub_on=None):
    pts = f"{x0},{y0} {x0 + W},{y0} {x0 + W + SKEW},{y0 + H} {x0 + SKEW},{y0 + H}"
    fill, edge, text, sw = (GO, INK, INK, 2.5) if on else (PALE, PALE_EDGE, PALE_TEXT, 1.5)
    cx = x0 + SKEW / 2 + W / 2
    out = [f'<polygon points="{pts}" fill="{fill}" stroke="{edge}" stroke-width="{sw}"/>']
    ty = y0 + (H / 2 + 6 if not sub else H / 2 - 2)
    out.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="19" font-style="italic" font-weight="600" '
               f'fill="{text}" text-decoration="{"underline" if on else "none"}">{escape(title)}</text>')
    if sub:
        parts = [f'<tspan font-weight="{700 if (on and s == sub_on) else 400}" '
                 f'fill="{INK if (on and (sub_on is None or s == sub_on)) else ("#3f6f4f" if on else PALE_TEXT)}">{s}</tspan>'
                 for s in ("random forest", "CNN")]
        out.append(f'<text x="{cx}" y="{y0 + H / 2 + 17}" text-anchor="middle" font-size="13">'
                   f'{parts[0]}<tspan fill="{PALE_TEXT}"> · </tspan>{parts[1]}</text>')
    return out


def here(x, y):
    return [f'<text x="{x}" y="{y}" font-size="14" font-weight="700" fill="{INK}">◀ you are here</text>']


def svg(active: list[str], sub_on: str | None, marker: bool = True) -> str:
    el = [f'<rect x="1" y="1" width="998" height="488" rx="18" fill="{MINT}" stroke="#cfeedd" stroke-width="2"/>',
          f'<text x="32" y="44" font-size="22" font-weight="600" fill="{INK}" text-decoration="underline">'
          f'Drip Detector · CI-Pipeline</text>']
    # Step 0 · camera
    el += [f'<rect x="40" y="70" width="58" height="40" rx="7" fill="{INK}"/>',
           f'<rect x="50" y="63" width="18" height="9" rx="2" fill="{INK}"/>',
           f'<circle cx="69" cy="90" r="13" fill="{GO}" stroke="#fff" stroke-width="2.5"/>',
           f'<polygon points="69,82 76,95 62,95" fill="{INK}"/>',
           f'<text x="112" y="96" font-size="15" fill="{INK}">Step 0 · snap on sdux.tech/computer-vision</text>']
    for lane, (lx, stages, label) in enumerate([(70, PRE, "Pre-Processing"), (520, PRO, "Processing")]):
        el.append(f'<circle cx="{lx - 12}" cy="{150}" r="5" fill="{INK}"/>')
        el.append(f'<text x="{lx}" y="156" font-size="20" font-weight="500" fill="{INK}">{label}</text>')
        el.append(f'<circle cx="{lx - 14}" cy="176" r="6" fill="none" stroke="{INK}" stroke-width="2.5"/>')
        el.append(f'<line x1="{lx - 12}" y1="182" x2="{lx + 2 * SW + 12}" y2="{176 + 3 * SH - 4}" '
                  f'stroke="{INK}" stroke-width="2.5" marker-end="url(#arrow)"/>')
        for i, name in enumerate(stages):
            x0, y0 = lx + i * SW, 170 + i * SH
            on = name in active
            el += shape(x0, y0, name, on, "yes" if name == "Classifying" else None, sub_on)
            if marker and on and "Inference" not in active:
                el += here(x0 + W + SKEW + 12, y0 + H / 2 + 5)
    # Inference · laptop
    on = "Inference" in active
    fill, edge = (GO, INK) if on else (PALE, PALE_EDGE)
    el += [f'<rect x="760" y="392" width="92" height="58" fill="{fill}" stroke="{edge}" stroke-width="4"/>',
           f'<rect x="750" y="450" width="112" height="7" rx="2" fill="{edge}"/>',
           f'<polygon points="806,402 826,440 786,440" fill="{INK if on else PALE_EDGE}"/>',
           f'<text x="742" y="428" text-anchor="end" font-size="17" font-weight="{600 if on else 400}" '
           f'fill="{INK if on else PALE_TEXT}">Inference</text>']
    if marker and on:
        el += here(868, 428)
    defs = (f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10" fill="none" stroke="{INK}" stroke-width="2"/></marker></defs>')
    title = "Drip Detector CI-Pipeline, highlighted: " + ", ".join(active)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 490" width="1000" height="490" role="img" '
            f'font-family="{FONT}"><title>{escape(title)}</title>{defs}{"".join(el)}</svg>\n')


if __name__ == "__main__":
    for topic, (active, sub_on) in TOPICS.items():
        (ROOT / topic).mkdir(exist_ok=True)
        (ROOT / topic / "pipeline.svg").write_text(svg(active, sub_on, marker=topic != "."), encoding="utf-8")
        print("wrote", topic + "/pipeline.svg")
