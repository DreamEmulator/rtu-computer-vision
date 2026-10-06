"""Get the template's updates into your project: link it once, then merge what's new, on a branch of its own.

    python -m tools.sync_template                 # link (the first time), fetch, merge on a new branch
    python -m tools.sync_template --check         # only show what's new in the template
    python -m tools.sync_template --url <git url> # a different template (default: the course template)

What it does:
  1. 🔗 adds the template as a git remote called "template" (once)
  2. 📥 fetches it and lists what's new since your last sync
  3. 🌿 creates a branch template-sync/<date> from where you are, so main isn't touched
  4. 🔀 merges the template into that branch. Your stage values, your code, your images and your project's
     name stay yours; where you and the template changed the same lines, git asks you to choose
  5. 🚀 tells you what to do next: run the pipeline, push the branch, open a pull request

A repository made with "Use this template" starts without the template's history, so git can't merge it
as is. The first sync finds the template commit your project started from and records it as already
merged: after that, only what changed in the template since then comes in.
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEMPLATE_URL = "https://github.com/DreamEmulator/rtu-computer-vision.git"
REMOTE = "template"


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise SystemExit(f"❌ git {' '.join(args)}\n{result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def same_repo(a: str, b: str) -> bool:
    def norm(url: str) -> str:
        url = url.strip().removesuffix(".git").removesuffix("/").lower()
        return url.replace("git@github.com:", "github.com/").split("://")[-1]
    return norm(a) == norm(b)


def link(url: str) -> None:
    """🔗 The template as a remote called "template"."""
    remotes = git("remote").split()
    if REMOTE not in remotes:
        git("remote", "add", REMOTE, url)
        print(f"🔗 Linked the template: git remote '{REMOTE}' → {url}")
    elif not same_repo(git("remote", "get-url", REMOTE), url):
        git("remote", "set-url", REMOTE, url)
        print(f"🔗 The template remote now points at {url}")


def starting_point(upstream: str) -> str | None:
    """The template commit this project was created from: the one whose files match this project's first commit."""
    roots = git("rev-list", "--max-parents=0", "HEAD").split()
    trees = {git("rev-parse", f"{r}^{{tree}}") for r in roots}
    for line in git("log", "--format=%H %T", upstream).splitlines():
        commit, tree = line.split()
        if tree in trees:
            return commit
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=TEMPLATE_URL, help="the template's git URL (default: the course template)")
    ap.add_argument("--branch", default="main", help="the template's branch to take updates from")
    ap.add_argument("--check", action="store_true", help="only show what's new; change nothing")
    a = ap.parse_args()
    upstream = f"{REMOTE}/{a.branch}"

    if git("rev-parse", "--is-inside-work-tree", check=False) != "true":
        raise SystemExit("❌ This isn't a git repository. Run it inside your project.")
    origin = git("remote", "get-url", "origin", check=False)
    if origin and same_repo(origin, a.url):
        raise SystemExit("👾 This IS the template: there's nothing to sync it with.")
    if not a.check and git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("❌ You have uncommitted changes. Commit them (or git stash) first, so nothing of yours gets lost.")

    # 1–2 · Link and fetch
    link(a.url)
    print(f"📥 Fetching {a.url} …")
    git("fetch", REMOTE, a.branch)

    # Linked before (a fork, or synced before)? Then git knows what we share. Otherwise: find the starting point.
    shared = git("merge-base", "HEAD", upstream, check=False)
    start = None
    if not shared:
        start = starting_point(upstream)
        if start is None:
            raise SystemExit("❌ I can't find the template commit this project started from (your first commit's files\n"
                             "   match none of the template's). Ask your lecturer; they can link it by hand.")
    new = git("log", "--format=  • %s (%cs)", f"{shared or start}..{upstream}").splitlines()
    if not new:
        print("✅ You're up to date: the template has nothing new since your last sync.")
        return
    print(f"🆕 {len(new)} update{'s' if len(new) != 1 else ''} in the template since your last sync:")
    print("\n".join(new[:30]) + (f"\n  … and {len(new) - 30} more" if len(new) > 30 else ""))
    if a.check:
        print("\n(--check: nothing changed. Run without --check to merge them.)")
        return

    # 3 · A branch of its own
    here = git("rev-parse", "--abbrev-ref", "HEAD")
    branch = f"template-sync/{datetime.date.today():%Y-%m-%d}"
    if git("rev-parse", "--verify", "--quiet", branch, check=False):
        branch += f"-{datetime.datetime.now():%H%M%S}"
    git("switch", "-c", branch)
    print(f"🌿 On a new branch: {branch} (from {here})")

    # 4 · Merge. The first time: record the starting point as merged, without changing a single file.
    if start:
        git("merge", "-s", "ours", "--allow-unrelated-histories", "--no-edit", "-m",
            f"🔗 Link to the template at {start[:7]}: everything up to here is already in this project", start)
        print(f"🔗 Linked your history to the template commit this project started from ({start[:7]})")
    merged = subprocess.run(["git", "merge", "--no-ff", "--no-edit", "-m", f"🔀 Updates from the template ({a.branch})", upstream],
                            cwd=ROOT, capture_output=True, text=True)
    if merged.returncode == 0:
        print("🔀 Merged without conflicts.")
        next_steps(branch, here)
        return

    conflicts = git("diff", "--name-only", "--diff-filter=U").splitlines()
    if not conflicts:
        raise SystemExit(f"❌ The merge stopped:\n{merged.stdout}{merged.stderr}")
    pictures = [c for c in conflicts if c.endswith("pipeline.svg")]
    if pictures:                                       # pictures are drawn, not written: redraw them with your name
        from tools.make_pipeline_svg import TOPICS, project_name, svg
        project = project_name()
        for folder, (active, sub_on) in TOPICS.items():
            path = f"{folder}/pipeline.svg" if folder != "." else "pipeline.svg"
            if path in pictures:
                (ROOT / path).write_text(svg(active, sub_on, marker=folder != ".", project=project), encoding="utf-8")
                git("add", path)
        conflicts = [c for c in conflicts if c not in pictures]
        print(f"🎨 Redrew {len(pictures)} pipeline picture{'s' if len(pictures) != 1 else ''} with your project's name.")
    if not conflicts:
        git("commit", "--no-edit")
        print("🔀 Merged.")
        next_steps(branch, here)
        return

    print(f"\n✋ You and the template changed the same lines in {len(conflicts)} file{'s' if len(conflicts) != 1 else ''}:")
    print("\n".join(f"  • {c}" for c in conflicts))
    print("\n   Choose, file by file:\n"
          "   • PyCharm: Git → Resolve Conflicts… (yours on the left, the template's on the right)\n"
          "   • or open the file and keep what you want between the <<<<<<< ======= >>>>>>> markers\n"
          "   Rules of thumb: in action.yaml keep YOUR stage values; in the stage code and the README take the\n"
          "   template's new parts and keep your own changes.\n"
          "   Then:  git add <file> …  and  git commit --no-edit\n"
          f"   Changed your mind? git merge --abort && git switch {here} && git branch -D {branch}")


def next_steps(branch: str, here: str) -> None:
    print("\n🚀 Next:\n"
          "   1. python run_pipeline.py              does everything still pass?\n"
          f"   2. git push -u origin {branch}\n"
          f"   3. open a pull request into {here}: CI runs on it, and you merge when it's green\n"
          f"   Not what you wanted? git switch {here} && git branch -D {branch}")


if __name__ == "__main__":
    main()
