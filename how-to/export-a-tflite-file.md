# ❓ How to export a tflite file from our pipeline?

> 🎯 **Goal** · a `model.tflite` that a phone app can run, the instructions for the frames it expects, and proof that both work.<br>
> 🗺️ **Route** · 🧠 6 Neural network → the `neural_network` artifact → your app<br>
> 🧰 **You need** · TensorFlow: `pip install -r requirements-cnn.txt` in PyCharm's Terminal

TensorFlow Lite (renamed *LiteRT* in 2024) runs a trained network on a phone, without Python. Only the 🧠 network can become a `.tflite` file; the 🌳 forest goes another way (see *If you see…*).

## 🪜 Steps

### 1 · Switch on the network and the export
- [ ] In the 🎛️ tinker zone of [`06_neural-networks/classifying_cnn/action.yaml`](../06_neural-networks/classifying_cnn/action.yaml):
  ```yaml
    enabled:
      default: "true"
    export:
      default: "tflite_quantized"
  ```
- [ ] Run it: `python run_pipeline.py --from neural_network` locally, or push and download the `neural_network` artifact from the Actions run.

You get three files in `build/neural_network/`:

| File | What it is |
|---|---|
| `model.tflite` | the network, flattened into one file |
| `labels.json` | which output is which class |
| `preprocessing.json` | everything stages 1–4 do to a frame before the network sees it, with every knob |

### 2 · Check the export, don't assume it
- [ ] `tflite agrees with keras %` should be 100, or very close. The stage runs the *file* on every test frame, the way the phone will.
- [ ] Compare `tflite size KB` for `"tflite"` and `"tflite_quantized"`. On our data: 1122 KB against 288 KB, with the same test accuracy.

### 3 · Find the catch
Save this as `try_tflite.py` in the repository root and run it:

```python
import json
import cv2
import numpy as np
import tensorflow as tf

labels = json.load(open("build/neural_network/labels.json"))
model = tf.lite.Interpreter(model_path="build/neural_network/model.tflite")
model.allocate_tensors()
inp, out = model.get_input_details()[0], model.get_output_details()[0]

def ask(frame, what):
    model.set_tensor(inp["index"], frame.reshape(inp["shape"]).astype(np.float32))
    model.invoke()
    p = model.get_tensor(out["index"])[0]
    print(f"{what:<40} → {labels[p.argmax()]} ({p.max():.0%})")

ask(cv2.imread("build/segmenting/images/snap.png", cv2.IMREAD_GRAYSCALE), "the snap after stages 1–4")
raw = cv2.imread("snaps/default_drip.jpg", cv2.IMREAD_GRAYSCALE)
ask(cv2.resize(raw, (128, 128), interpolation=cv2.INTER_AREA), "the raw photo, only resized")
```

What we got:

```
the snap after stages 1–4                → drop (100%)
the raw photo, only resized              → no_drop (100%)
```

The same model and the same photo give opposite answers, and both times it's sure. The network only understands frames that went through stages 1–4 exactly as in training. A phone that feeds it raw camera frames gets confident nonsense.

### 4 · Choose how the phone gets those frames
- [ ] Open `preprocessing.json` and pick one way out:

| | How | Price |
|---|---|---|
| **A** · port it | rebuild stages 1–4 in the app. OpenCV runs on Android and iOS, and each stage's code (the `recipe.py` for Segmenting) is your spec, one `cv2` call at a time | work for the app developer, and two copies that must stay identical |
| **B** · bake it in | move preprocessing *into* the network as layers: resizing and rescaling are easy, adaptive thresholds and contours aren't | you may have to simplify stages 1–4 first |
| **C** · simplify it | `pass_on: "original"` in Segmenting, `method: "none"` in Cleaning and Improving, then measure what accuracy that costs | maybe a few % of accuracy, for an app that's far easier to build |

- [ ] Run option C once, whichever you choose. Now you know what stages 2–4 are worth on your data.

## 🔀 If you see…

| You see | It means | Try |
|---|---|---|
| ❌ `The neural network needs TensorFlow` | it isn't installed in this Python | `pip install -r requirements-cnn.txt` |
| pip can't find a TensorFlow version | your Python is newer than TensorFlow supports | a `.venv` with Python 3.11, the version CI uses |
| `tflite agrees with keras` below 100 after quantization | 8-bit weights are rounded, so a few borderline frames flip | fine as long as `tflite test accuracy` still passes the gates |
| the export works, but the gates are red | a tflite file of a bad model is a bad model on a phone | ship nothing; [How to pick which model I am training?](pick-which-model-i-am-training.md) |
| you want the 🌳 forest on the phone | TFLite is for neural networks | ONNX: `skl2onnx` converts scikit-learn models (beyond this How To) |

## 🏆 Done when
- `model.tflite`, `labels.json` and `preprocessing.json` downloaded from a green CI run.
- `tflite agrees with keras %` is 100 or very close.
- You can explain to an app developer, with `preprocessing.json` open, which of A, B or C you chose and why.

## 💼 At work this is called…
- **Edge deployment**: running the model on the device instead of a server (TFLite / LiteRT, Core ML, ONNX Runtime).
- **Post-training quantization**: smaller and faster by storing weights as 8-bit integers.
- **Training–serving skew**: the model sees different data in production than in training. Google's *Rules of Machine Learning* treat it as one of the most common ways deployed models fail.
- **An interface contract**: `preprocessing.json` is one. The model and the app agree on it, in writing.

⬅️ [All How To's](README.md) · [The whole pipeline](../README.md)
