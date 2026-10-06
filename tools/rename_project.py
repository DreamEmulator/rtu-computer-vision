"""Give the template your project's name (and pitch). The Drip Detector is the worked example it ships with.

    python -m tools.rename_project "Cow Counter"
    python -m tools.rename_project "Cow Counter" --pitch "A barn camera that counts the cows at milking time."
    python -m tools.rename_project "Cow Counter" --dry-run          # only show what would change

It changes the places that say what THIS project is called, and nothing else:

    7_demo-day/action.yaml          project_name (the one source of the name) and, with --pitch, the pitch
    README.md                       the title
    <stage>/README.md               the caption of every pipeline picture
    <stage>/pipeline.svg            the pipeline pictures, redrawn with the new name
    .github/workflows/pipeline.yml  the header
    .devcontainer/devcontainer.json the Codespace's name

The example's story stays as it is: drip chambers in the stage READMEs, the synthetic drip dataset, the
question every stage answers. Rewrite those for your product when you're ready; your own images go in
images_for_training/ (see how-to/add-my-own-classes.md). Run it again any time to rename again.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXAMPLE = "Drip Detector"


def current_name() -> str:
    from stages.common import stage_values_of
    return str(stage_values_of("demo_day").get("project_name") or EXAMPLE)


def plan(new: str, pitch: str | None) -> dict[Path, str]:
    """→ {file: its new text} for every file that would change."""
    olds = sorted({current_name(), EXAMPLE}, key=len, reverse=True)
    any_old = "|".join(re.escape(o) for o in olds)
    rules: list[tuple[Path, str, str]] = [
        (ROOT / "7_demo-day" / "action.yaml", r'(project_name:\n(?:[ \t]+description:.*\n)?[ \t]+default: )"[^"\n]*"', rf'\g<1>"{new}"'),
        (ROOT / "README.md", rf"\A(# \S+ )(?:{any_old})( — )", rf"\g<1>{new}\g<2>"),
        (ROOT / ".github" / "workflows" / "pipeline.yml", rf"(#  🚀 )(?:{'|'.join(re.escape(o.upper()) for o in olds)})( — CI-PIPELINE)",
         rf"\g<1>{new.upper()}\g<2>"),
        (ROOT / ".devcontainer" / "devcontainer.json", rf'("name": ")(?:{any_old})( — CV pipeline")', rf"\g<1>{new}\g<2>"),
    ]
    if pitch is not None:
        rules.append((ROOT / "7_demo-day" / "action.yaml", r'(pitch:\n(?:[ \t]+description:.*\n)?[ \t]+default: )"[^"\n]*"', rf'\g<1>"{pitch}"'))
    rules += [(readme, rf'(alt="The )(?:{any_old})( CI-pipeline with )', rf"\g<1>{new}\g<2>")
              for readme in sorted(ROOT.glob("[1-7]_*/README.md"))]

    changed: dict[Path, str] = {}
    for path, pattern, replacement in rules:
        if not path.exists():
            continue
        text = changed.get(path, path.read_text(encoding="utf-8"))
        text = re.sub(pattern, replacement, text, flags=re.M)   # names can't hold " or \\ (see check)
        if text != path.read_text(encoding="utf-8"):
            changed[path] = text
    return changed


def check(text: str, what: str) -> str:
    text = text.strip()
    if not text or len(text) > 80 or any(c in text for c in '"\\\n'):
        raise SystemExit(f'❌ The {what} must be 1–80 characters, on one line, without " or \\.')
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help='your project\'s name, e.g. "Cow Counter"')
    ap.add_argument("--pitch", help="one sentence an investor remembers (shown on Demo Day)")
    ap.add_argument("--dry-run", action="store_true", help="only show what would change")
    a = ap.parse_args()
    new, pitch = check(a.name, "name"), (check(a.pitch, "pitch") if a.pitch else None)
    old = current_name()

    changes = plan(new, pitch)
    for path in changes:
        print(f"{'would change' if a.dry_run else '✏️  changed'}  {path.relative_to(ROOT)}")
    if a.dry_run:
        print(f"…and redraw every pipeline.svg with “{new}”. Nothing was written (--dry-run).")
        raise SystemExit(0)
    for path, text in changes.items():
        path.write_text(text, encoding="utf-8")

    from tools.make_pipeline_svg import TOPICS, svg
    for folder, (active, sub_on) in TOPICS.items():
        (ROOT / folder / "pipeline.svg").write_text(svg(active, sub_on, marker=folder != ".", project=new), encoding="utf-8")
    print(f"🎨 redrew {len(TOPICS)} pipeline pictures")
    print(f"\n✅ “{old}” is now “{new}”." + ("" if pitch else " Add your pitch with --pitch, or in 7_demo-day/action.yaml."))
    print("   Next: your own images in images_for_training/ (how-to/add-my-own-classes.md). The stories in the stage READMEs\n"
          "   are still the Drip Detector example: rewrite them for your product when you're ready.")
