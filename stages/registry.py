"""The pipeline map: every stage, the week's topic it belongs to, and where its action lives.

Standard library only, so the webhook CLI can use it before any dependency is installed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    key: str            # build/<key>/ locally, artifact <key> in CI
    n: int              # position on the Drip Detector slide
    title: str
    action: str         # <stage>_<action> folder holding action.yaml
    topic: str          # /<topic> folder
    topic_title: str    # topic of the week
    emoji: str
    lecture: str
    question: str       # the startup question this stage answers
    previous: str | None
    notebook: str | None
    todo: str

    @property
    def action_path(self) -> str:
        return f"{self.topic}/{self.action}"


STAGES = [
    Stage("digital_data", 1, "Digital Data", "digital-data_prepare", "01_image-processing",
          "Introduction to Image Processing", "📷", "Tue 22.09",
          "Can we turn whatever the camera gives us into one consistent format?",
          None, "digital-data.ipynb", "Todo_Digital-Data.md"),
    Stage("cleaning", 2, "Cleaning", "cleaning_denoise", "02_image-preprocessing",
          "Image Preprocessing Methods", "🧽", "Thu 24.09",
          "Our clip-on camera is cheap. Can we trust its pixels?",
          "digital_data", "cleaning.ipynb", "Todo_Cleaning.md"),
    Stage("improving", 3, "Improving", "improving_enhance", "02_image-preprocessing",
          "Image Preprocessing Methods", "🔆", "Thu 24.09",
          "Night shift on the ward: will we still see anything in dim light?",
          "cleaning", "improving.ipynb", "Todo_Improving.md"),
    Stage("segmenting", 4, "Segmenting", "segmenting_threshold", "03_image-segmentation",
          "Image Segmentation", "✂️", "Tue 29.09",
          "Which pixels belong to the drip chamber, and which are just background?",
          "improving", "segmenting.ipynb", "Todo_Segmenting.md"),
    Stage("extracting", 5, "Extracting", "extracting_describe", "04_feature-extraction",
          "Feature Extraction and Data Preparation", "🧬", "Thu 01.10",
          "How do we describe a frame with numbers a model can learn from?",
          "segmenting", "extracting.ipynb", "Todo_Extracting.md"),
    Stage("random_forest", 6, "Classifying · Random Forest", "classifying_random-forest", "05_random-forests",
          "Image Classification with Random Forests", "🌳", "Tue 06.10",
          "Is the infusion running, stopped, or about to run dry?",
          "extracting", "random-forest.ipynb", "Todo_Classifying.md"),
    Stage("neural_network", 6, "Classifying · Neural Network", "classifying_cnn", "06_neural-networks",
          "Image Classification with Neural Networks", "🧠", "Thu 08.10",
          "Can a network learn its own features, straight from the pixels?",
          "extracting", "neural-network.ipynb", "Todo_Classifying.md"),
    Stage("demo_day", 7, "Demo Day", "demo-day_report", "07_course-summary",
          "Course Summary", "🎤", "Tue 13.10",
          "What do we show the investors (and the hospital)?",
          None, None, "Todo_Demo-Day.md"),
]
BY_KEY = {s.key: s for s in STAGES}
ORDER = [s.key for s in STAGES]
