"""Step 0 from the command line: send a photo URL through your pipeline on GitHub, like the sdux.tech page does.

    python -m tools.snap https://example.com/my-drip.jpg --label drop
    python -m tools.snap https://example.com/my-drip.jpg --webhook https://sdux.tech/api/cv/webhook

Needs the GitHub CLI (gh), logged in with access to your repository. The photo must be reachable by URL
(the page uploads it first); for a local file use:  python run_pipeline.py --snap photo.jpg
"""
from __future__ import annotations

import argparse
import subprocess
import uuid

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--label", default="")
    ap.add_argument("--webhook", default="")
    ap.add_argument("--ref", default="main")
    a = ap.parse_args()
    snap_id = uuid.uuid4().hex[:12]
    cmd = ["gh", "workflow", "run", "pipeline.yml", "--ref", a.ref, "-f", f"snap_url={a.url}", "-f", f"snap_id={snap_id}"]
    if a.label:
        cmd += ["-f", f"snap_label={a.label}"]
    if a.webhook:
        cmd += ["-f", f"webhook_url={a.webhook}"]
    subprocess.run(cmd, check=True)
    print(f"📸 Snap {snap_id} is on its way. Watch it with:  gh run watch")
