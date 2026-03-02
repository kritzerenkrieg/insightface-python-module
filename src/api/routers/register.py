from fastapi import APIRouter, HTTPException, UploadFile, File
from face_recog.register import register_image

router = APIRouter()

@router.post("/register")
async def register(name: str, image: UploadFile = File(...), db: str = "./db"):
    try:
        contents = await image.read()
        out = register_image(contents, name, db)
        return {"message": "Registered", "data": out}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))