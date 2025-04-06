from transformers import pipeline
import threading

_classifier = None
_lock = threading.Lock()

def get_classifier():
    global _classifier
    if _classifier is None:
        with _lock:
            if _classifier is None:
                _classifier = pipeline("text-classification", model="cointegrated/rubert-tiny-toxicity", top_k=None)
    return _classifier

def predict_toxicity_text(text):
    classifier = get_classifier()
    predictions = classifier(text)
    return [{"label": item["label"], "score": round(item["score"], 4)} for item in predictions[0]]
