import argparse
from fastapi import FastAPI, HTTPException
from face_recog.register import register_image
from face_recog.recognize import recognize_image
from pydantic import BaseModel

app = FastAPI()

class RegisterRequest(BaseModel):
    image: str
    name: str
    db: str = "./db"

class RecognizeRequest(BaseModel):
    image: str
    db: str = "./db"
    topk: int = 1
    threshold: float = 0.5

@app.post("/register")
async def register(request: RegisterRequest):
    out = register_image(request.image, request.name, request.db)
    return {"message": "Registered", "output": out}

@app.post("/recognize")
async def recognize(request: RecognizeRequest):
    res = recognize_image(request.image, request.db, top_k=request.topk, threshold=request.threshold)
    if not res:
        raise HTTPException(status_code=404, detail="No matches found")
    return res