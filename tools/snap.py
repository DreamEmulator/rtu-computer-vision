"""Step 0 from the command line: send a photo URL through your pipeline on GitHub.

    python -m tools.snap https://example.com/my-drip.jpg

The same as Actions → 🚀 CI-Pipeline → Run workflow with the URL pasted in. Needs the GitHub CLI (gh),
logged in with access to your repository. For a local file use:  python run_pipeline.py --snap photo.jpg
"""
from __future__ import annotations

import argparse
import subprocess

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--ref", default="main", help="the branch whose stage values the photo runs through")
    a = ap.parse_args()
    subprocess.run(["gh", "workflow", "run", "pipeline.yml", "--ref", a.ref, "-f", f"snap_url={a.url}"], check=True)
    print("📸 Your image for analyzing is on its way. Watch it with:  gh run watch")
