from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.text_model import predict_toxicity_text

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict-text")
def predict_text(input: TextInput):
    result = predict_toxicity_text(input.text)
    return result
