from transformers import pipeline

# Ensure compatibility with latest transformers versions
classifier = pipeline("text-classification", model="unitary/toxic-bert")

def detect_toxic_text(text):
    result = classifier(text)
    return result
