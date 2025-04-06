from transformers import pipeline

# Load the toxic comment classification model
classifier = pipeline("text-classification", model="unitary/toxic-bert")

def detect_toxic_text(text: str):
    """
    Classifies the input text using unitary/toxic-bert and returns top predictions.

    Args:
        text (str): The input text to analyze.

    Returns:
        list: A list of dictionaries with labels and scores.
    """
    result = classifier(text, top_k=None)  # Return all class scores
    cleaned_result = [
        {"label": item["label"].lower().replace("toxicity_", ""), "score": round(item["score"], 4)}
        for item in result
    ]
    return cleaned_result
