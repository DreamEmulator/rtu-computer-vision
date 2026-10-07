# ❓ How to pick which model I am training?

> 🎯 **Goal** · choose the classifier that ships, the random forest or the neural network, with numbers, and say what it costs.<br>
> 🗺️ **Route** · 🧬 5 Extracting → 🌳 6 random forest and 🧠 6 neural network → 🎤 7 Demo Day<br>
> 🧰 **You need** · the pipeline running; the synthetic data is fine. For the network: `pip install -r requirements-cnn.txt`

## The candidates

Both run side by side after Extracting, sit the same model evaluation, and Demo Day shows them next to each other.

| | Where | Learns from |
|---|---|---|
| 🌳 random forest | [`6_random-forest/`](../6_random-forest/): always on | stage 5's feature vectors (HOG, LBP): numbers *you* chose to describe each image |
| 🧠 neural network | [`6_neural-network/`](../6_neural-network/): `enabled: "true"` | stage 4's pixels: it learns its own filters and ignores stage 5's features |

## 🪜 Steps

### 1 · Know the bar
- [ ] Every run reports `baseline (always guess the majority)`: 0.33 with three balanced classes. A model that doesn't clearly beat it hasn't learned anything.

### 2 · Run both on the same data
- [ ] The forest: `python run_pipeline.py --from random_forest` (stages 1–5 stay as they are).
- [ ] The network: `enabled: "true"` in its `action.yaml`, then `python run_pipeline.py --from neural_network`.
- [ ] Fill in the table from both reports.

Our numbers on the synthetic day-zero data, on a laptop (yours will differ, and that's the point):

| | test accuracy | worst-class recall | training s | ms per prediction | model size |
|---|---|---|---|---|---|
| 🌳 random forest | **0.98** | **0.93** drop | **1.5** | **0.3** | **363 KB** |
| 🧠 neural network, default values | 0.78 | 0.57 drop | 13 | 1.3 | 3.4 MB · 288 KB as quantized tflite |

### 3 · Weigh it up

| | 🌳 Random forest | 🧠 Neural network |
|---|---|---|
| **Images you need** | dozens to hundreds per class | hundreds to thousands per class; with fewer it memorises |
| **Can you explain a decision?** | yes: the report names the features it relies on | hard: hundreds of thousands of learned weights |
| **Your effort goes into…** | good features (stage 5) | data and training (epochs, dropout, augmentation) |
| **When the classes differ subtly** | limited by your features | can find patterns nobody thought to describe |
| **On a phone** | the app must also compute HOG and LBP | tflite is built for phones; the app still needs stages 1–4 |

### 4 · Decide, and write it down
- [ ] One sentence: *"We ship ___ because ___, even though ___."*
- [ ] Shipping the forest? Set the network back to `enabled: "false"`: a network that fails its gates turns CI red, even when the forest passes.

## 🔀 If you see…

| You see | It means | Try |
|---|---|---|
| network: training accuracy near 100 %, the frames that sit out far below | memorising: too many weights for too few images | `dropout: "0.5"`, `head: "gap"`, `augment_copies: "3"`, more data. See its [README](../6_neural-network/README.md) |
| cv accuracy far from test accuracy | a small test set: one lucky or unlucky frame moves the score a lot | trust the cv number more, and collect more data |
| both models within a few % of each other | the data or the features are the ceiling, not the model | [How to train our model on cows?](train-our-model-on-cows.md) |
| the network wins on your own images | your classes differ in ways HOG and LBP don't describe | keep it, and check the memorising gap once more |

## 🏆 Done when
A filled-in table, one chosen model, and the sentence from step 4 in your pull request.

## 💼 At work this is called…
- **Model selection** against a **baseline**: always report what "doing nothing clever" scores.
- **No free lunch** (Wolpert, 1996): no model wins on every problem, so you measure on yours.
- **Latency and size budgets**: a clip-on camera has a small CPU and a battery, so ms per prediction and KB are requirements, not trivia.
- **Occam's razor**: when the accuracy is equal, ship the smaller, faster, simpler model.

⬅️ [All How To's](README.md) · [The whole pipeline](../README.md)
