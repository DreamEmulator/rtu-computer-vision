# Step 0 · Snap — the contract for sdux.tech/computer-vision

> **Not in use right now.** Students send a snap by pasting a photo URL into **Actions → 🚀 CI-Pipeline →
> Run workflow** (or `python -m tools.snap <url>`), and read the result in Demo Day's report on GitHub Pages.
> This contract stays here for a page that wants to start runs and follow their webhooks.

The page does three things: **upload** a photo, **start** the student's pipeline with it, and **show** the
webhooks the pipeline sends back while it runs. This document is everything the page needs to know.

```
 📱 page ──upload──▶ sdux backend ──dispatch──▶ GitHub Actions (student's repo)
    ▲                    │   ▲                        │
    └──── poll ──────────┘   └──── signed webhooks ───┘  pipeline.started, stage.started,
                                                          stage.finished ×7, pipeline.finished
```

## 1 · Upload
Store the photo where GitHub's runners can download it (any public or pre-signed URL). Stage 1 fetches it
with a 30 s timeout and accepts anything OpenCV decodes (jpg, png, webp …) up to 15 MB. Resize on the phone
first if you like; the pipeline shrinks it to 128 × 128 anyway.

## 2 · Start the pipeline
A page uses `repository_dispatch`: it's the only call that carries `snap_id`, `snap_label` and `webhook_url`.
The Run workflow form (`workflow_dispatch`) takes `snap_url` and nothing else, so the students see one field.

```http
POST https://api.github.com/repos/{owner}/{repo}/dispatches        (needs Contents: write)
Authorization: Bearer <installation token>
Accept: application/vnd.github+json

{"event_type": "snap",
 "client_payload": {"snap_url": "…", "snap_id": "a1b2c3d4", "snap_label": "drop", "webhook_url": "…"}}
```

| input | required | meaning |
|---|---|---|
| `snap_url` | yes | where the runner downloads the photo |
| `snap_id` | recommended | echoed in **every** event, so the page can match events to the upload (GitHub doesn't hand back a run id on dispatch) |
| `snap_label` | no | what the student says it shows: `drop`, `no_drop`, `low_fluid`. Enables `correct: true/false` in the verdict |
| `webhook_url` | no | overrides the repository variable `CV_WEBHOOK_URL` for this run |

**Tokens.** Don't ask students for personal access tokens. A small GitHub App ("SDUX Computer Vision") with the
permission *Contents: read & write* (for `repository_dispatch`), installed by each student on their repository, gives your backend a
short-lived installation token per repo. `repo=owner/name` arrives in the page URL: every Todo links to
`https://sdux.tech/computer-vision?repo=<owner>/<name>`.

## 3 · Receive the webhooks
Every event is a `POST` of `application/json` to the webhook URL, with these headers:

| header | value |
|---|---|
| `X-CV-Event` | `pipeline.started` · `stage.started` · `stage.finished` · `pipeline.finished` |
| `X-CV-Signature-256` | `sha256=` + HMAC-SHA256 of the raw body, keyed with the repository secret `CV_WEBHOOK_SECRET` (only sent when the secret is set) |

Verify before you trust (Python; any language has HMAC):

```python
expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
ok = hmac.compare_digest(expected, request.headers["X-CV-Signature-256"])
```

Answer with any 2xx within 10 s. Failed deliveries are logged as warnings in the run and **never** retried or
fatal. Store every event keyed by `snap_id` (plus `delivery` to deduplicate) and let the page poll your
backend, e.g. `GET /api/cv/snaps/a1b2c3d4/events?after=<delivery>` every second or two.

Every event carries: `event`, `delivery` (uuid), `sent_at`, `snap_id`, `repository`, `run_id`, `run_attempt`,
`run_url`, `sha`, `ref`, `trigger`.

### Order of events
`pipeline.started` → for each stage `stage.started` then `stage.finished` → `pipeline.finished`.
Random forest and neural network run in parallel after Extracting, so their events interleave. If a gate
fails, the stages after it never start: expect no events for them, and `pipeline.finished` lists them as
`skipped`.

### `stage.finished` → `status`
| status | meaning | show |
|---|---|---|
| `passed` | all gates green | metrics, snap images |
| `failed` | a gate failed; the pipeline stops after this stage | `gates[]` where `ok` is false |
| `off` | switched off in its action.yaml (the CNN until 08.10) | a grey card |
| `error` | the stage crashed | `error.message` (and `error.traceback` for the curious) |

`snap.images` holds data-URI PNGs of the photo at this stage (upscaled with sharp pixels to ≥ 256 px):
Digital Data `camera`, `output` · Cleaning `output`, `removed` · Improving `output`, `edges` ·
Segmenting `overlay`, `output` · Extracting `hog`, `lbp`. The classifiers send their verdict in
`snap.metrics`: `prediction`, `confidence`, `probabilities`, and with a label `you said`, `correct`.
`performance` has milliseconds per frame for every stage; Digital Data adds the runner's SIMD instruction
sets and its number of CUDA GPUs.

### Example: `pipeline.started`
```json
{
  "event": "pipeline.started",
  "delivery": "80fe6cde-e33c-4192-bee1-3b818a4bbc5c",
  "sent_at": "2026-09-27T12:00:52Z",
  "snap_id": "a1b2c3d4",
  "repository": "student/drip-detector",
  "run_id": "13824511",
  "run_attempt": "1",
  "run_url": "https://github.com/student/drip-detector/actions/runs/13824511",
  "sha": "4f2c9e1…",
  "ref": "main",
  "trigger": "workflow_dispatch",
  "stages": [
    {
      "key": "digital_data",
      "n": 1,
      "title": "Digital Data",
      "emoji": "📷",
      "topic": "Introduction to Image Processing",
      "action": "1_digital-data",
      "lecture": "Tue 22.09"
    },
    {
      "key": "cleaning",
      "n": 2,
      "title": "Cleaning",
      "emoji": "🧽",
      "topic": "Image Preprocessing Methods",
      "action": "2_cleaning",
      "lecture": "Thu 24.09"
    },
    "…"
  ],
  "snap": {
    "url": "/tmp/does-not-exist.jpg",
    "label": null
  }
}
```

### Example: `stage.finished` (Cleaning)
```json
{
  "event": "stage.finished",
  "delivery": "e9c4aefa-a1b2-46ae-9f17-0bb19ff6f7b4",
  "sent_at": "2026-09-27T12:00:22Z",
  "snap_id": "a1b2c3d4",
  "repository": "student/drip-detector",
  "run_id": "13824511",
  "run_attempt": "1",
  "run_url": "https://github.com/student/drip-detector/actions/runs/13824511",
  "sha": "4f2c9e1…",
  "ref": "main",
  "trigger": "workflow_dispatch",
  "stage": {
    "key": "cleaning",
    "n": 2,
    "title": "Cleaning",
    "emoji": "🧽",
    "topic": "Image Preprocessing Methods",
    "action": "2_cleaning",
    "lecture": "Thu 24.09"
  },
  "status": "passed",
  "duration_s": 0.96,
  "knobs": {
    "method": "median",
    "kernel": 3,
    "sigma": 0,
    "…": "…"
  },
  "metrics": {
    "method": "median",
    "noise σ before": 5.5938,
    "noise σ after": 0.8441,
    "sharpness before": 1398.2916,
    "sharpness after": 295.3425
  },
  "performance": {
    "ms per frame": 0.0093
  },
  "gates": [
    {
      "name": "noise σ after cleaning",
      "value": 0.8441,
      "rule": "≤ 4.0",
      "ok": true
    }
  ],
  "snap": {
    "metrics": {
      "noise σ before": 2.0682,
      "noise σ after": 0.6797
    },
    "images": {
      "output": "data:image/png;base64,iVBORw0KGgo…",
      "removed": "data:image/png;base64,iVBORw0KGgo…"
    }
  },
  "error": null
}
```

### Example: `stage.finished` (Random forest, with the verdict)
```json
{
  "event": "stage.finished",
  "delivery": "5c3973fc-e934-4369-b44c-fec2b32dbdd4",
  "sent_at": "2026-09-27T12:00:40Z",
  "snap_id": "a1b2c3d4",
  "repository": "student/drip-detector",
  "run_id": "13824511",
  "run_attempt": "1",
  "run_url": "https://github.com/student/drip-detector/actions/runs/13824511",
  "sha": "4f2c9e1…",
  "ref": "main",
  "trigger": "workflow_dispatch",
  "stage": {
    "key": "random_forest",
    "n": 6,
    "title": "Classifying · Random Forest",
    "emoji": "🌳",
    "topic": "Image Classification with Random Forests",
    "action": "6_random-forest",
    "lecture": "Tue 06.10"
  },
  "status": "passed",
  "duration_s": 7.35,
  "knobs": {
    "model": "random_forest",
    "n_estimators": 200,
    "max_depth": null,
    "…": "…"
  },
  "metrics": {
    "cv accuracy (mean ± std)": "0.907 ± 0.010",
    "model": "random_forest",
    "baseline (always guess the majority)": 0.3333,
    "test accuracy": 0.9667,
    "precision (macro)": 0.9697,
    "recall (macro)": 0.9667,
    "f1 (macro)": 0.9666,
    "recall per class": {
      "drop": 0.9,
      "low_fluid": 1.0,
      "no_drop": 1.0
    },
    "confusion matrix (rows = true)": "27 0 3 / 0 30 0 / 0 0 30"
  },
  "performance": {
    "training seconds": 7.0738,
    "ms per prediction": 0.1372,
    "model size KB": 365.7705
  },
  "gates": [
    {
      "name": "test accuracy",
      "value": 0.9667,
      "rule": "≥ 0.85",
      "ok": true
    },
    {
      "name": "worst class recall",
      "value": 0.9,
      "rule": "≥ 0.75",
      "ok": true
    }
  ],
  "snap": {
    "metrics": {
      "prediction": "drop",
      "confidence": 0.495,
      "probabilities": {
        "drop": 0.495,
        "low_fluid": 0.04,
        "no_drop": 0.465
      },
      "you said": "drop",
      "correct": true
    },
    "images": {}
  },
  "error": null
}
```

### Example: `pipeline.finished`
```json
{
  "event": "pipeline.finished",
  "delivery": "f69e2b51-4f2a-4bf0-b630-5697cf877d14",
  "sent_at": "2026-09-27T12:00:40Z",
  "snap_id": "a1b2c3d4",
  "repository": "student/drip-detector",
  "run_id": "13824511",
  "run_attempt": "1",
  "run_url": "https://github.com/student/drip-detector/actions/runs/13824511",
  "sha": "4f2c9e1…",
  "ref": "main",
  "trigger": "workflow_dispatch",
  "status": "passed",
  "stages": [
    {
      "key": "digital_data",
      "title": "Digital Data",
      "status": "passed",
      "duration_s": 1.56
    },
    {
      "key": "cleaning",
      "title": "Cleaning",
      "status": "passed",
      "duration_s": 0.96
    },
    {
      "key": "improving",
      "title": "Improving",
      "status": "passed",
      "duration_s": 1.01
    },
    {
      "key": "segmenting",
      "title": "Segmenting",
      "status": "passed",
      "duration_s": 1.5
    },
    {
      "key": "extracting",
      "title": "Extracting",
      "status": "passed",
      "duration_s": 7.85
    },
    {
      "key": "random_forest",
      "title": "Classifying · Random Forest",
      "status": "passed",
      "duration_s": 7.35
    },
    {
      "key": "neural_network",
      "title": "Classifying · Neural Network",
      "status": "off",
      "duration_s": 0.0
    }
  ],
  "predictions": {
    "random_forest": {
      "prediction": "drop",
      "confidence": 0.495,
      "probabilities": {
        "drop": 0.495,
        "low_fluid": 0.04,
        "no_drop": 0.465
      },
      "you said": "drop",
      "correct": true
    }
  },
  "total_s": 20.23
}
```

## Setting up a student repository
1. Repository **variable** `CV_WEBHOOK_URL` = your webhook endpoint. Then push and pull-request runs report to
   the page too, not only snaps. (Settings → Secrets and variables → Actions → Variables.)
2. Repository **secret** `CV_WEBHOOK_SECRET` = the shared secret for signatures. One per student is best; the
   GitHub App can't set secrets with *Actions* permission alone, so either let the page show the student a
   secret to paste, or give the App *Secrets: write* as well.
3. Install the GitHub App on the repository.

## Developing the page without GitHub
```bash
python -m tools.webhook_receiver --port 8765 --secret s3cret            # terminal 1: prints + logs events
CV_WEBHOOK_URL=http://localhost:8765 CV_WEBHOOK_SECRET=s3cret \
SNAP_ID=a1b2c3d4 python run_pipeline.py --snap photo.jpg --snap-label drop   # terminal 2
```
The receiver appends every event to `webhook-events.jsonl`: real payloads to build the UI against.
From a laptop with the GitHub CLI: `python -m tools.snap https://…/photo.jpg --label drop --webhook https://…`.
