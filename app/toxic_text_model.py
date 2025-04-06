from transformers import pipeline

# Lightweight model for toxicity detection
classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

def detect_toxic_text(text):
    result = classifier(text)
    return result
