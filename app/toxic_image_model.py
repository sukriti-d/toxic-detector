from nsfw_detector import predict
from PIL import Image
import threading
import os

_model = None
_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                model_path = "nsfw_model"  # Downloaded automatically
                _model = predict.load_model(model_path)
    return _model

def predict_toxicity_image(image: Image.Image):
    model = get_model()
    predictions = model.predict(image)
    scores = list(predictions.values())[0]  # Get scores dict

    return {
        "categories": scores,
        "verdict": "toxic" if scores["porn"] > 0.4 or scores["hentai"] > 0.4 else "safe"
    }
