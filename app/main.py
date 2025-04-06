from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from app.toxic_text_model import predict_toxicity_text
from app.toxic_image_model import predict_toxicity_image
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for text input
class TextData(BaseModel):
    text: str

@app.post("/predict-text")
async def predict_text(data: TextData):
    try:
        result = predict_toxicity_text(data.text)
        return {"success": True, "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text prediction error: {str(e)}")

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predict_toxicity_image(image)
        return {"success": True, "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image prediction error: {str(e)}")
