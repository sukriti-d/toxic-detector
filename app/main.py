from fastapi import FastAPI, Request
from app.text_model import predict_toxicity_text

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Text toxicity detection API is running."}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}
    score = predict_toxicity_text(text)
    return {"toxicity_score": score}
