"""Step 0's other half: tell the sdux.tech page what happens to the snap, while it happens.

Every event is a JSON POST to $CV_WEBHOOK_URL. When $CV_WEBHOOK_SECRET is set, the body is signed
with HMAC-SHA256 in the header  X-CV-Signature-256: sha256=<hex>  (GitHub's own webhook scheme).
No URL, no events. A webhook that can't be delivered never fails the pipeline.

Events: pipeline.started · stage.started · stage.finished · pipeline.finished
Contract and examples: docs/step-0-snap.md

CLI, used by the composite actions before Python dependencies are installed (stdlib only):
    python3 -m stages.webhook started --stage cleaning
    python3 -m stages.webhook crashed --stage cleaning
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

from stages.registry import BY_KEY, ORDER

ROOT = Path(__file__).resolve().parents[1]


def context() -> dict:
    env = os.environ.get
    repo, run_id = env("GITHUB_REPOSITORY"), env("GITHUB_RUN_ID")
    return {
        "snap_id": env("SNAP_ID") or None,
        "repository": repo or "local",
        "run_id": run_id,
        "run_attempt": env("GITHUB_RUN_ATTEMPT"),
        "run_url": f"{env('GITHUB_SERVER_URL', 'https://github.com')}/{repo}/actions/runs/{run_id}" if repo and run_id else None,
        "sha": env("GITHUB_SHA"),
        "ref": env("GITHUB_HEAD_REF") or env("GITHUB_REF_NAME"),
        "trigger": env("GITHUB_EVENT_NAME", "local"),
    }


def stage_info(key: str) -> dict:
    s = BY_KEY[key]
    return {"key": s.key, "n": s.n, "title": s.title, "emoji": s.emoji, "topic": s.topic_title,
            "action": s.action_path, "lecture": s.lecture}


def emit(event: str, payload: dict) -> bool:
    url = os.environ.get("CV_WEBHOOK_URL", "").strip()
    if not url:
        return False
    body = json.dumps({"event": event, "delivery": str(uuid.uuid4()),
                       "sent_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       **context(), **payload}, default=str).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "cv-startup-pipeline", "X-CV-Event": event}
    secret = os.environ.get("CV_WEBHOOK_SECRET", "")
    if secret:
        headers["X-CV-Signature-256"] = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return 200 <= r.status < 300
    except Exception as e:  # the page being down must never break the pipeline
        print(f"::warning title=Webhook not delivered::{event} → {url}: {e}")
        return False


def started(key: str) -> None:
    if key == ORDER[0]:
        emit("pipeline.started", {"stages": [stage_info(k) for k in ORDER],
                                  "snap": {"url": os.environ.get("SNAP_URL") or None,
                                           "label": os.environ.get("SNAP_LABEL") or None}})
    emit("stage.started", {"stage": stage_info(key)})


def crashed(key: str) -> None:
    """Only speaks up when the stage itself never got to report (e.g. pip install failed)."""
    out = ROOT / "build" / key
    if (out / "metrics.json").exists() or (out / "error.json").exists():
        return
    emit("stage.finished", {"stage": stage_info(key), "status": "error",
                            "error": {"message": "The job failed before this stage could report. Open the run log.",
                                      "traceback": None}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["started", "crashed"])
    ap.add_argument("--stage", required=True, choices=ORDER)
    a = ap.parse_args()
    {"started": started, "crashed": crashed}[a.what](a.stage)
