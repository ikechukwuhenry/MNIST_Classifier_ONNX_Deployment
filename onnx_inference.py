import onnxruntime as ort
import numpy as np
import PIL.Image as Image
import os

# Load ONNX model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
onnx_model_path = os.path.join(BASE_DIR, 'deployment/mnist_model.onnx')
session = ort.InferenceSession(onnx_model_path)

# prepare input data (example: a single MNIST image)
image_path = os.path.join(BASE_DIR, "sample_images/img_10.jpg")   # Replace with your image path
image = Image.open(image_path).convert('L')                     # Convert to grayscale

# Resize and normalize the image to match the training preprocessing
image = image.resize((28, 28))
image_array = np.array(image).astype(np.float32) / 255.0    # Normalize to [0, 1]
image_array = (image_array - 0.5) / 0.2                     # same normalization as training
image_array = image_array.reshape(1, 1, 28, 28)             # Reshape to (batch_size, channels, height, width)

# dummy_input = np.random.rand(1, 1, 28, 28).astype(np.float32)

# Run inference
outputs = session.run(None, {"images": image_array})
predicted_class = np.argmax(outputs[0], axis=1)
print("Predicted class:", predicted_class[0])
print("ONNX model output:", outputs[0])


