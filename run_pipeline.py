"""Run the pipeline locally — the same stages CI runs, in the same order, with the knobs from the action.yaml files.

    python run_pipeline.py                              # everything
    python run_pipeline.py --snap ~/Desktop/drip.jpg    # Step 0: send your own photo through every stage
    python run_pipeline.py --from segmenting            # re-run from a stage, reusing earlier outputs
    python run_pipeline.py --to 3                       # stop after stage 3 (keys or slide numbers both work)
"""
from __future__ import annotations

import argparse
import importlib
import os
import time

from stages.registry import BY_KEY, ORDER

MODULES = {"digital_data": "stages.s1_digital_data", "cleaning": "stages.s2_cleaning", "improving": "stages.s3_improving",
           "segmenting": "stages.s4_segmenting", "extracting": "stages.s5_extracting",
           "random_forest": "stages.s6_classifying", "neural_network": "stages.s6_classifying",
           "demo_day": "stages.s7_demo_day"}


def _index(which: str | int | None, default: int, last: bool = False) -> int:
    if which is None:
        return default
    if str(which).isdigit():
        hits = [i for i, k in enumerate(ORDER) if BY_KEY[k].n == int(which)]
        return hits[-1] if last else hits[0]
    return ORDER.index(str(which))


def run(start=None, stop=None, keep_going: bool = False) -> bool:
    """Returns True when every stage that ran passed its gates."""
    from stages.deps import check
    check()  # a friendly message instead of a traceback when this Python lacks a package
    from stages import webhook
    from stages.common import run_stage
    ok = True
    for key in ORDER[_index(start, 0):_index(stop, len(ORDER) - 1, last=True) + 1]:
        module = importlib.import_module(MODULES[key])
        main = (lambda k=key: module.main(k)) if MODULES[key].endswith("classifying") else module.main
        t0 = time.time()
        webhook.started(key)  # in CI the composite action sends this before installing anything
        try:
            run_stage(key, main)
        except SystemExit as e:
            if e.code not in (0, None):
                ok = False
                print(f"\n⛔ {BY_KEY[key].title}: {e.code if isinstance(e.code, str) else 'a quality gate did not pass'}")
                if not keep_going:
                    return False
        print(f"   ⏱  {time.time() - t0:.1f}s")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="stop")
    ap.add_argument("--snap", help="path or URL of a photo to send through every stage (Step 0)")
    ap.add_argument("--snap-label", help="what you think the photo shows: drop | no_drop | low_fluid")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed gate (like CI's Demo Day)")
    a = ap.parse_args()
    if a.snap:
        os.environ["SNAP_URL"] = a.snap
    if a.snap_label:
        os.environ["SNAP_LABEL"] = a.snap_label
    raise SystemExit(0 if run(a.start, a.stop, a.keep_going) else 1)
