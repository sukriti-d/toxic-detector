from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("tomh/toxigen-roberta-tiny")
model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen-roberta-tiny")

def predict_toxicity_text(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        toxic_score = scores[0][1].item()
    return toxic_score
