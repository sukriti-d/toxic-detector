import onnxruntime as ort
from PIL import Image
import numpy as np
import os

# Load model once
_model = None

def load_model():
    global _model
    if _model is None:
        _model = ort.InferenceSession("lite_nsfw.onnx")  # Place the file in root or app folder
    return _model

def preprocess(image: Image.Image):
    image = image.resize((224, 224)).convert("RGB")
    img_array = np.asarray(image).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1).reshape(1, 3, 224, 224)
    return img_array

LABELS = ["drawings", "hentai", "neutral", "porn", "sexy"]

def predict_toxicity_image(image: Image.Image):
    model = load_model()
    img = preprocess(image)
    inputs = {model.get_inputs()[0].name: img}
    outputs = model.run(None, inputs)[0][0]
    result = dict(zip(LABELS, [round(float(i), 4) for i in outputs]))
    verdict = "toxic" if result["porn"] > 0.4 or result["hentai"] > 0.4 else "safe"
    return {"categories": result, "verdict": verdict}
