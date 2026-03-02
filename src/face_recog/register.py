from fastapi import APIRouter, HTTPException
from face_recog.register import register_image
from pydantic import BaseModel

router = APIRouter()

class RegisterRequest(BaseModel):
    image: str
    name: str
    db: str = "./db"

@router.post("/register")
async def register(request: RegisterRequest):
    try:
        out = register_image(request.image, request.name, request.db)
        return {"message": "Registered", "data": out}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))