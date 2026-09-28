"""A tiny stand-in for the sdux.tech page: receive the pipeline's webhooks and show them.

    python -m tools.webhook_receiver --port 8765 --secret s3cret
    CV_WEBHOOK_URL=http://localhost:8765 CV_WEBHOOK_SECRET=s3cret python run_pipeline.py --snap photo.jpg

Every event is checked against its X-CV-Signature-256 header, printed as one line, and appended to
webhook-events.jsonl. The page's own backend should do the same: verify, store, and let the page poll.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def verify(secret: str, body: bytes, header: str | None) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header or "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--secret", default="")
    ap.add_argument("--log", default="webhook-events.jsonl")
    a = ap.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if a.secret and not verify(a.secret, body, self.headers.get("X-CV-Signature-256")):
                self.send_response(401); self.end_headers(); print("✗ bad signature"); return
            e = json.loads(body)
            stage = (e.get("stage") or {}).get("title", "")
            snap = (e.get("snap") or {}).get("metrics") or e.get("predictions") or ""
            print(f"{e['event']:<18} {stage:<30} {e.get('status', ''):<8} {e.get('duration_s', '')!s:<6} {snap}")
            with open(a.log, "a", encoding="utf-8") as f:
                f.write(json.dumps(e) + "\n")
            self.send_response(204); self.end_headers()

        def log_message(self, *args):
            pass

    print(f"Listening on http://localhost:{a.port} — events go to {a.log}")
    HTTPServer(("", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
