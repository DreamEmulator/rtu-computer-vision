# ❓ How to pick which model I am training?

> 🎯 **Goal** · choose the classifier that ships, with numbers, and say what it costs.<br>
> 🗺️ **Route** · 🧬 5 Extracting → 🌳 6 forest / SVM / kNN and 🧠 6 network → 🎤 7 Demo Day<br>
> 🧰 **You need** · the pipeline running; the synthetic data is fine. For the network: `pip install -r requirements-cnn.txt`

## The candidates

Two stage-6 jobs run side by side after Extracting, and Demo Day compares them.

| | Switch it on in | Learns from |
|---|---|---|
| 🌳 random forest | [`classifying_random-forest/action.yaml`](../05_random-forests/classifying_random-forest/action.yaml) → `model: "random_forest"` | stage 5's feature vectors (HOG, LBP, …), which only describe brightness |
| 📏 SVM | the same file → `model: "svm"` | the same vectors |
| 👥 kNN | the same file → `model: "knn"` | the same vectors |
| 🧠 CNN | [`classifying_cnn/action.yaml`](../06_neural-networks/classifying_cnn/action.yaml) → `enabled: "true"` | stage 4's pixels, every channel. It learns its own features and ignores stage 5's |

The forest job holds one model at a time; the network runs next to it.

## 🪜 Steps

### 1 · Know the bar
- [ ] Every run reports `baseline (always guess the majority)`: 0.33 with three balanced classes. A model that doesn't clearly beat it hasn't learned anything.

### 2 · Run every candidate on the same data
- [ ] Change `model`, then `python run_pipeline.py --from random_forest` (stages 1–5 stay as they are).
- [ ] For the network: `enabled: "true"`, then `python run_pipeline.py --from neural_network`.
- [ ] Fill in the table from each run's output.

Our numbers on the synthetic day-zero data, on a laptop (yours will differ, and that's the point):

| | test accuracy | worst-class recall | cv accuracy | training s | ms per prediction | model size KB |
|---|---|---|---|---|---|---|
| 🌳 random forest | **0.98** | 0.93 drop | 0.90 | 1.5 | 0.32 | **363** |
| 📏 SVM | 0.94 | 0.87 drop | 0.86 | 2.1 | 0.38 | 4114 |
| 👥 kNN | 0.79 | 0.43 drop | 0.69 | **0.13** | **0.07** | 3555 |
| 🧠 CNN, default knobs | 0.78 | 0.57 drop | — | 13 | 1.3 | 3406 · 288 as quantized tflite |

### 3 · Decide, and write it down
- [ ] One sentence: *"We ship ___ because ___, even though ___."*

## 🔀 If you see…

| You see | It means | Try |
|---|---|---|
| SVM or kNN far below the forest | both measure **distances**, so a feature with big numbers drowns the rest. Trees don't care. | check `scaling: "standard"` in stage 5. Without it, SVM drops from 0.94 to 0.88 and kNN from 0.79 to 0.73 on our data |
| cv accuracy far from test accuracy | a small test set: one lucky or unlucky frame moves the score a lot | trust the cv number more, and collect more data |
| CNN: training accuracy near 100 %, validation far below | overfitting: it memorised its training frames | `dropout: "0.5"`, `augment_copies: "3"`, more data. See its [Todo](../06_neural-networks/Todo_Classifying.md) |
| every model within a few % of each other | the data or the features are the ceiling, not the model | [How to train our model on cows?](train-our-model-on-cows.md) |
| kNN's model size grows with every frame you add | kNN keeps the whole training set: the data *is* the model | fine for 300 frames, not for 300 000 |

## 🏆 Done when
A filled-in table, one chosen model, and the sentence from step 3 in your pull request.

## 💼 At work this is called…
- **Model selection** against a **baseline**: always report what "doing nothing clever" scores.
- **No free lunch** (Wolpert, 1996): no model wins on every problem, so you measure on yours.
- **Latency and size budgets**: a clip-on camera has a small CPU and a battery, so ms per prediction and KB are requirements, not trivia.
- **Occam's razor**: when the accuracy is equal, ship the smaller, faster, simpler model.

⬅️ [All How To's](README.md) · [The whole pipeline](../README.md)
