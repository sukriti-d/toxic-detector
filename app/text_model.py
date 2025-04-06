from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load once
tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

LABELS = ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"]

def predict_toxicity_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.sigmoid(logits).squeeze().tolist()

    result = {label: round(prob, 4) for label, prob in zip(LABELS, probs)}
    verdict = "toxic" if any(prob > 0.5 for prob in probs) else "clean"
    return {"labels": result, "verdict": verdict}
