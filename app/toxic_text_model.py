from transformers import pipeline

# Lightweight model for toxicity detection
classifier = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-toxic-comments", top_k=None)

def detect_toxic_text(text):
    result = classifier(text)
    return result
