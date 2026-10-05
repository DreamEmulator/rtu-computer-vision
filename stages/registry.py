"""The pipeline map: every stage, its folder, the lecture it belongs to, and the file CI runs.

Standard library only, so the webhook CLI can use it before any dependency is installed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    key: str            # build/<key>/ locally, artifact <key> in CI
    n: int              # position on the Drip Detector slide
    title: str
    folder: str         # the stage's folder: README.md, action.yaml and its code
    code: str           # the Python file in that folder that CI runs
    topic_title: str    # the lecture's topic
    emoji: str
    lecture: str
    question: str       # the startup question this stage answers
    previous: str | None

    @property
    def action_path(self) -> str:   # CI imports the folder as a composite action: uses: ./<folder>
        return self.folder


STAGES = [
    Stage("digital_data", 1, "Digital Data", "1_digital-data", "digital_data.py",
          "Introduction to Image Processing", "📷", "Tue 22.09",
          "Can we turn whatever the camera gives us into one consistent format?", None),
    Stage("cleaning", 2, "Cleaning", "2_cleaning", "cleaning.py",
          "Image Preprocessing Methods", "🧽", "Thu 24.09",
          "Our clip-on camera is cheap. Can we trust its pixels?", "digital_data"),
    Stage("improving", 3, "Improving", "3_improving", "improving.py",
          "Image Preprocessing Methods", "🔆", "Thu 24.09",
          "Night shift on the ward: will we still see anything in dim light?", "cleaning"),
    Stage("segmenting", 4, "Segmenting", "4_segmenting", "segmenting.py",
          "Image Segmentation", "✂️", "Tue 29.09",
          "Which pixels belong to the drip chamber, and which are just background?", "improving"),
    Stage("extracting", 5, "Extracting", "5_extracting", "extracting.py",
          "Feature Extraction and Data Preparation", "🧬", "Thu 01.10",
          "How do we describe a frame with numbers a model can learn from?", "segmenting"),
    Stage("random_forest", 6, "Classifying · Random Forest", "6_random-forest", "random_forest.py",
          "Image Classification with Random Forests", "🌳", "Tue 06.10",
          "Is the infusion running, stopped, or about to run dry?", "extracting"),
    Stage("neural_network", 6, "Classifying · Neural Network", "6_neural-network", "neural_network.py",
          "Image Classification with Neural Networks", "🧠", "Thu 08.10",
          "Can a network learn its own features, straight from the pixels?", "extracting"),
    Stage("demo_day", 7, "Demo Day", "7_demo-day", "demo_day.py",
          "Course Summary", "🎤", "Tue 13.10",
          "What do we show the investors (and the hospital)?", None),
]
BY_KEY = {s.key: s for s in STAGES}
ORDER = [s.key for s in STAGES]
