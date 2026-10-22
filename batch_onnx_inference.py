import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
import os


# Load onnx
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
onnx_model_path = os.path.join(BASE_DIR, 'deployment/mnist_cnn_model.onnx')
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name      # "images"

def preprocess(path, invert=False):
    img = Image.open(path).convert("L")
    if invert:                                  # use if the digit is dark on a white background
        img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.2                     # same normalization as training
    return arr[np.newaxis, :, :]                # shape (1, 28, 28)

paths = [os.path.join(BASE_DIR, f"sample_images/img_{i}.jpg")
         for i in [9, 10, 11, 12, 18, 25, 26, 29, 35]]

# Stack into a batch: (N, 28, 28) -> add the channel dim -> (N, 1, 28, 28)
batch = np.stack([preprocess(p) for p in paths], axis=0)
print(batch.shape)                              # (9, 1, 28, 28)

# One inference call for the whole batch
logits = session.run(None, {input_name: batch})[0]    # shape (9, 10)

# Convert logits to probabilities (softmax)
exp = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = exp / exp.sum(axis=1, keepdims=True)

preds = probs.argmax(axis=1)
for path, pred, p in zip(paths, preds, probs.max(axis=1)):
    print(f"{path} -> {pred} ({p:.1%})")