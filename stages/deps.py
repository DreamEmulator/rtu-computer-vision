"""Is every package the pipeline needs installed in the Python that runs it? Standard library only."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PACKAGES = {"cv2": "opencv-python-headless", "numpy": "numpy", "yaml": "pyyaml", "sklearn": "scikit-learn",
            "skimage": "scikit-image", "matplotlib": "matplotlib", "joblib": "joblib"}   # import name: pip name


def check() -> None:
    missing = [pip for module, pip in PACKAGES.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    raise SystemExit(
        f"👾 This Python can't run the pipeline yet: it's missing {', '.join(missing)}.\n"
        f"   It runs from: {sys.executable}\n\n"
        f"   Install what the pipeline needs into it:\n"
        f'       "{sys.executable}" -m pip install -r "{requirements}"\n\n'
        f"   PyCharm: better give this project its own interpreter. Settings → Project → Python Interpreter →\n"
        f"   Add Interpreter → Add Local Interpreter → Virtualenv, location: .venv in this folder.\n"
        f"   Then open requirements.txt and click 'Install requirements', or run the line above in PyCharm's Terminal.")
