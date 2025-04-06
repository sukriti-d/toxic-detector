from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from app.toxic_text_model import detect_toxic_text
from app.toxic_image_model import detect_nsfw_image
import shutil, uuid, os

app = FastAPI()
API_KEY = os.getenv("API_KEY", "my-default-key")

def validate_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/analyze/text")
async def analyze_text(text: str = Form(...), x_api_key: str = Header(...)):
    validate_api_key(x_api_key)
    result = detect_toxic_text(text)
    return {"toxicity": result}

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...), x_api_key: str = Header(...)):
    validate_api_key(x_api_key)
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = detect_nsfw_image(temp_filename)
    os.remove(temp_filename)
    return {"nsfw_score": result}
